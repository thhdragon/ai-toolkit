import { spawn, execFile, ChildProcess } from 'child_process';
import { promisify } from 'util';
import os from 'os';
import si from 'systeminformation';
import { loadMacstats } from '@/server/macstats';
import { createLoadSampler, readCpuTemperature, readMemory } from '@/server/cpuStats';
import { CpuInfo, GpuInfo, GPUApiResponse, MonitorHistoryPoint, MonitorInit, MonitorSample } from '@/types';
import { historyPointFromSample, MONITOR_HISTORY_LENGTH, MONITOR_TICK_MS } from '@/utils/monitorSample';

const execFileAsync = promisify(execFile);

/**
 * Always-on system monitor. Samples CPU + GPU every MONITOR_TICK_MS, keeps a
 * 2-minute rolling history (load + memory only), and pushes every full sample
 * to subscribers (the SSE route at /api/monitor).
 *
 * GPU stats come from a single resident `nvidia-smi ... -lms` child process —
 * NVML stays initialized between samples, which is what made the old
 * spawn-per-request /api/gpu route slow. If loop mode never produces output
 * (or nvidia-smi keeps hanging), we fall back to a one-shot spawn per tick,
 * which matches the old route's behavior exactly.
 */

const NV_QUERY_ARGS = [
  '--query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed',
  '--format=csv,noheader,nounits',
];
const NV_ENV = { ...process.env, CUDA_DEVICE_ORDER: 'PCI_BUS_ID' };

export const ROCM_QUERY_ARGS = [
  '--showproductname',
  '--showtemp',
  '--showuse',
  '--showmemuse',
  '--showpower',
  '--showmeminfo',
  'vram',
  '--showdriverversion',
  '--showclocks',
  '--showfan',
  '--showid',
  '--json',
];

export function parseRocmOutput(stdout: string): GpuInfo[] {
  let data: Record<string, any>;
  try {
    data = JSON.parse(stdout.trim());
  } catch {
    return [];
  }

  const driverVersion = data.system?.['Driver version'] || 'ROCm';
  const gpus: GpuInfo[] = [];

  for (const [key, cardVal] of Object.entries(data)) {
    if (!key.startsWith('card') || typeof cardVal !== 'object' || !cardVal) continue;
    const card = cardVal as Record<string, any>;
    const indexMatch = key.match(/^card(\d+)$/);
    const index = indexMatch ? parseInt(indexMatch[1], 10) : 0;

    const name = card['Card Series'] || card['Device Name'] || `AMD Radeon (Card ${index})`;

    let temp = 0;
    const tempKey = Object.keys(card).find(k => k.toLowerCase().includes('temperature'));
    if (tempKey && card[tempKey]) {
      temp = Math.round(parseFloat(card[tempKey])) || 0;
    }

    const gpuUtil = parseInt(card['GPU use (%)'] ?? card['GPU use'], 10) || 0;
    let memUtil = parseInt(card['GPU Memory Allocated (VRAM%)'], 10);

    const totalBytes = parseFloat(card['VRAM Total Memory (B)']) || 0;
    const usedBytes = parseFloat(card['VRAM Total Used Memory (B)']) || 0;
    const totalMb = Math.round(totalBytes / (1024 * 1024));
    const usedMb = Math.round(usedBytes / (1024 * 1024));
    const freeMb = Math.max(0, totalMb - usedMb);

    if (isNaN(memUtil) || memUtil < 0) {
      memUtil = totalMb > 0 ? Math.round((usedMb / totalMb) * 100) : 0;
    }

    let powerDraw = 0;
    const powerKey = Object.keys(card).find(
      k => k.includes('Current Socket Graphics Package Power') || k.includes('Power'),
    );
    if (powerKey && card[powerKey]) {
      powerDraw = parseFloat(card[powerKey]) || 0;
    }
    const powerLimit = parseFloat(card['Max Graphics Package Power (W)'] || card['Power Cap (W)']) || 0;

    let clockGraphics = 0;
    if (card['sclk clock speed:']) {
      const match = String(card['sclk clock speed:']).match(/(\d+)/);
      if (match) clockGraphics = parseInt(match[1], 10);
    }
    let clockMemory = 0;
    if (card['mclk clock speed:']) {
      const match = String(card['mclk clock speed:']).match(/(\d+)/);
      if (match) clockMemory = parseInt(match[1], 10);
    }

    let fanSpeed = 0;
    const fanKey = Object.keys(card).find(k => k.toLowerCase().includes('fan speed'));
    if (fanKey && card[fanKey]) {
      fanSpeed = parseInt(card[fanKey], 10) || 0;
    }

    gpus.push({
      index,
      name,
      driverVersion,
      temperature: temp,
      utilization: {
        gpu: gpuUtil,
        memory: memUtil,
      },
      memory: {
        total: totalMb,
        free: freeMb,
        used: usedMb,
      },
      power: {
        draw: powerDraw,
        limit: powerLimit,
      },
      clocks: {
        graphics: clockGraphics,
        memory: clockMemory,
      },
      fan: {
        speed: fanSpeed,
      },
    });
  }

  return gpus.sort((a, b) => a.index - b.index);
}
// No stdout line for this long means the loop child is buffering or hung.
const NV_WATCHDOG_MS = 15_000;
// A line-less gap this long after the last line closes out a batch (all
// lines of one iteration arrive together; iterations are MONITOR_TICK_MS
// apart, so this can never bleed into the next batch).
const NV_BATCH_FLUSH_MS = 100;
// Temperature refresh is decoupled from the tick (see refreshCpuTemp)
const CPU_TEMP_REFRESH_MS = 5000;

// Power limits for GPUs whose driver reports none (nvidia-smi gives [N/A]).
// These unified SoCs have no settable GPU limit, so show the rated TDP of the
// whole chip. GB10 (DGX Spark) is 140 W. RTX Spark N1X: the 5120-core part is
// laptop-only, rated 45-80 W (OEM configurable); the 6144-core part is 140 W
// on desktop and 45-80 W in laptops but shares its name, so it falls through
// to the desktop figure.
const KNOWN_POWER_LIMITS_W: [RegExp, number][] = [
  [/\bGB10\b/, 140],
  [/RTX Spark.*5120-core/, 80],
  [/RTX Spark/, 140],
];

function knownPowerLimit(name: string): number {
  return KNOWN_POWER_LIMITS_W.find(([re]) => re.test(name))?.[1] ?? NaN;
}

function parseGpuLine(line: string): GpuInfo | null {
  const [
    index,
    name,
    driverVersion,
    temperature,
    gpuUtil,
    memoryUtil,
    memoryTotal,
    memoryFree,
    memoryUsed,
    powerDraw,
    powerLimit,
    clockGraphics,
    clockMemory,
    fanSpeed,
  ] = line.split(', ').map(item => item.trim());
  if (isNaN(parseInt(index))) return null;
  return {
    index: parseInt(index),
    name,
    driverVersion,
    temperature: parseInt(temperature),
    utilization: {
      gpu: parseInt(gpuUtil),
      memory: parseInt(memoryUtil),
    },
    memory: {
      total: parseInt(memoryTotal),
      free: parseInt(memoryFree),
      used: parseInt(memoryUsed),
    },
    power: {
      draw: parseFloat(powerDraw),
      limit: isNaN(parseFloat(powerLimit)) ? knownPowerLimit(name) : parseFloat(powerLimit),
    },
    clocks: {
      graphics: parseInt(clockGraphics),
      memory: parseInt(clockMemory),
    },
    fan: {
      speed: parseInt(fanSpeed) || 0, // Some GPUs might not report fan speed, default to 0
    },
  };
}

// Integrated GPUs that share system RAM with the CPU.
const UNIFIED_MEMORY_GPU = /\b(GB10|RTX Spark|Radeon\s+[0-9]+[A-Z]?\s+Graphics|Radeon\s+Graphics)\b/i;

/**
 * Unified-memory GPUs (GB10 / DGX Spark, RTX Spark, AMD APUs) have no dedicated VRAM
 * or only a small carveout (512MB-1GB). Report the system RAM pool they share instead.
 */
function isSharedMemoryGpu(g: GpuInfo): boolean {
  return isNaN(g.memory.total) || g.memory.total <= 1024 || UNIFIED_MEMORY_GPU.test(g.name);
}

function withSharedMemory(gpu: GPUApiResponse, cpu: CpuInfo | null): GPUApiResponse {
  if (!cpu || !gpu.gpus.some(isSharedMemoryGpu)) return gpu;
  const total = Math.round(cpu.totalMemory);
  const free = Math.round(cpu.availableMemory);
  return {
    ...gpu,
    gpus: gpu.gpus.map(g =>
      isSharedMemoryGpu(g) ? { ...g, memory: { total, free, used: total - free, shared: true } } : g,
    ),
  };
}

// Serialized once per tick and shared, so N subscribers don't stringify N times
type Subscriber = (sample: MonitorSample, serialized: string) => void;

class SystemMonitor {
  private isMac = os.platform() === 'darwin';
  private history: MonitorHistoryPoint[] = [];
  private subscribers = new Set<Subscriber>();
  private latestCpu: CpuInfo | null = null;
  private latestGpu: GPUApiResponse = { hasNvidiaSmi: false, hasRocmSmi: false, isMac: this.isMac, gpus: [] };
  private macGpuName = 'Apple GPU';
  private gpuBackend: 'detecting' | 'nvidia' | 'rocm' | 'none' = 'detecting';
  private rocmUnavailable = false;
  private rocmInFlight = false;
  private nvChild: ChildProcess | null = null;
  private nvBatch: GpuInfo[] = [];
  private nvStdoutBuffer = '';
  private nvFlushTimer: NodeJS.Timeout | null = null;
  private nvUnavailable = false;
  private nvReaped = false;
  private nvEverGotLine = false;
  private nvOneShotMode = false;
  private nvOneShotInFlight = false;
  private lastNvLineAt = 0;
  private tickInFlight = false;
  private lastCpuTemp = 0;
  private cpuTempInFlight = false;
  private lastCpuTempAt = 0;
  private cpuStatic: { name: string; cores: number } | null = null;
  // Differences os.cpus() times between ticks; no child process involved.
  private sampleLoad = createLoadSampler();

  start(): void {
    if (this.isMac) {
      this.initMacGpuName();
    } else {
      this.detectAndStartGpu();
    }
    // Never leave a resident nvidia-smi behind.
    process.once('exit', () => {
      try {
        this.nvChild?.kill('SIGKILL');
      } catch {
        // already gone
      }
    });
    // Fixed cadence; a tick that overruns the interval (slow temperature
    // read, hung nvidia-smi one-shot) just skips beats instead of stacking.
    this.tick();
    setInterval(() => this.tick(), MONITOR_TICK_MS);
  }

  subscribe(fn: Subscriber): () => void {
    this.subscribers.add(fn);
    return () => this.subscribers.delete(fn);
  }

  getInit(): MonitorInit {
    return {
      t: Date.now(),
      cpu: this.latestCpu,
      gpu: withSharedMemory(this.latestGpu, this.latestCpu),
      history: [...this.history],
    };
  }

  // -------------------------------------------------------------------------
  // Tick loop
  // -------------------------------------------------------------------------
  private async tick(): Promise<void> {
    if (this.tickInFlight) return;
    this.tickInFlight = true;
    try {
      await this.doTick();
    } finally {
      this.tickInFlight = false;
    }
  }

  private async doTick(): Promise<void> {
    const t = Date.now();
    try {
      this.latestCpu = await this.sampleCpu();
    } catch (error) {
      console.error('Monitor: CPU sample failed:', error);
    }
    try {
      if (this.isMac) {
        this.latestGpu = this.sampleMacGpu();
      } else if (this.gpuBackend === 'rocm') {
        await this.sampleRocmOneShot();
      } else if (this.gpuBackend === 'nvidia') {
        if (this.nvOneShotMode) {
          await this.sampleNvOneShot();
        } else {
          this.nvWatchdog();
        }
      }
    } catch (error) {
      console.error('Monitor: GPU sample failed:', error);
    }

    const sample: MonitorSample = {
      t,
      cpu: this.latestCpu,
      gpu: withSharedMemory(this.latestGpu, this.latestCpu),
    };
    this.history.push(historyPointFromSample(sample));
    if (this.history.length > MONITOR_HISTORY_LENGTH) {
      this.history.splice(0, this.history.length - MONITOR_HISTORY_LENGTH);
    }
    const serialized = JSON.stringify(sample);
    for (const subscriber of this.subscribers) {
      try {
        subscriber(sample, serialized);
      } catch (error) {
        console.error('Monitor: subscriber failed:', error);
      }
    }
  }

  // -------------------------------------------------------------------------
  // CPU (mirrors /api/cpu exactly)
  // -------------------------------------------------------------------------
  private async sampleCpu(): Promise<CpuInfo> {
    // si.cpu() shells out to lscpu/dmidecode on every call — at tick cadence
    // that alone was eating a measurable slice of a core. Name and core count
    // never change, so resolve them exactly once.
    if (!this.cpuStatic) {
      const cpuInfoRaw = await si.cpu();
      this.cpuStatic = { name: `${cpuInfoRaw.manufacturer} ${cpuInfoRaw.brand}`, cores: cpuInfoRaw.cores };
    }
    const cpuStatic = this.cpuStatic;

    if (this.isMac) {
      try {
        const ms = loadMacstats();
        if (!ms) throw new Error('macstats unavailable');
        const ramData = ms.getRAMUsageSync();
        const cpuData = ms.getCpuDataSync();
        return {
          name: cpuStatic.name,
          cores: cpuStatic.cores,
          temperature: cpuData.temperature || 0,
          totalMemory: ramData.total / (1024 * 1024),
          availableMemory: ramData.free / (1024 * 1024),
          freeMemory: ramData.free / (1024 * 1024),
          currentLoad: this.sampleLoad(),
        };
      } catch {
        // Fallback to the generic path if macstats fails
      }
    }

    // The temperature read can take >1s on some machines and would make the
    // tick skip beats, so it refreshes concurrently and we use the cached
    // value (at most a tick or two old).
    this.refreshCpuTemp();
    // Nothing below spawns a process. Every child process forks this server
    // first, and forking a multi-GB Node heap freezes the event loop for
    // hundreds of ms — at tick cadence that stalled every request in the UI.
    const memoryData = await readMemory();
    return {
      name: cpuStatic.name,
      cores: cpuStatic.cores,
      temperature: this.lastCpuTemp,
      totalMemory: memoryData.total / (1024 * 1024),
      availableMemory: memoryData.available / (1024 * 1024),
      freeMemory: memoryData.free / (1024 * 1024),
      currentLoad: this.sampleLoad(),
    };
  }

  private refreshCpuTemp(): void {
    // On Linux this is a sysfs read; elsewhere it may still shell out (via
    // systeminformation), and e.g. `sensors` can take ~1s of slow SMBus
    // polling — refreshed back-to-back it becomes a permanently resident
    // child. Every few seconds is plenty for a temperature.
    if (this.cpuTempInFlight || Date.now() - this.lastCpuTempAt < CPU_TEMP_REFRESH_MS) return;
    this.lastCpuTempAt = Date.now();
    this.cpuTempInFlight = true;
    readCpuTemperature()
      .then(t => {
        this.lastCpuTemp = t ?? 0;
      })
      .catch(() => {
        // keep the previous value
      })
      .finally(() => {
        this.cpuTempInFlight = false;
      });
  }

  // -------------------------------------------------------------------------
  // Mac GPU (mirrors /api/gpu's mac path)
  // -------------------------------------------------------------------------
  private initMacGpuName(): void {
    execFileAsync('sh', ['-c', 'system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset Model|Total Number of Cores"'], {
      timeout: 5000,
    })
      .then(({ stdout }) => {
        const nameMatch = stdout.match(/Chipset Model:\s*(.+)/);
        const coresMatch = stdout.match(/Total Number of Cores:\s*(\d+)/);
        if (nameMatch) {
          this.macGpuName = nameMatch[1].trim();
          if (coresMatch) {
            this.macGpuName += ` GPU (${coresMatch[1]} cores)`;
          }
        }
      })
      .catch(() => {
        // fallback to generic name
      });
  }

  private sampleMacGpu(): GPUApiResponse {
    let temperature = 0;
    let gpuLoad = 0;
    let fanSpeed = 0;
    let powerDraw = 0;
    let memUsed = 0;
    let memTotal = os.totalmem() / (1024 * 1024);

    const ms = loadMacstats();
    if (ms) {
      try {
        const gpuData = ms.getGpuDataSync();
        temperature = gpuData.temperature || 0;
        gpuLoad = gpuData.usage || 0;
      } catch {
        // ignore
      }
      try {
        const fanData = ms.getFanDataSync();
        const fanKeys = Object.keys(fanData);
        if (fanKeys.length > 0) {
          fanSpeed = fanData[fanKeys[0]].rpm || 0;
        }
      } catch {
        // ignore
      }
      try {
        const powerData = ms.getPowerDataSync();
        powerDraw = powerData.gpu || 0;
      } catch {
        // ignore
      }
      try {
        const ramData = ms.getRAMUsageSync();
        memUsed = ramData.used / (1024 * 1024);
        memTotal = ramData.total / (1024 * 1024);
      } catch {
        // ignore
      }
    }

    return {
      hasNvidiaSmi: false,
      isMac: true,
      gpus: [
        {
          index: 0,
          name: this.macGpuName,
          driverVersion: 'macOS',
          temperature: Math.round(temperature),
          utilization: {
            gpu: gpuLoad,
            memory: memTotal > 0 ? Math.round((memUsed / memTotal) * 100) : 0,
          },
          memory: {
            total: Math.round(memTotal),
            free: Math.round(memTotal - memUsed),
            used: Math.round(memUsed),
          },
          power: { draw: powerDraw, limit: 0 },
          clocks: { graphics: 0, memory: 0 },
          fan: { speed: fanSpeed },
        },
      ],
    };
  }

  // -------------------------------------------------------------------------
  // GPU backend detection & ROCm sampling
  // -------------------------------------------------------------------------
  private async detectAndStartGpu(): Promise<void> {
    const isWindows = process.platform === 'win32';
    let hasNv = false;
    try {
      const checkCmd = isWindows ? 'where nvidia-smi' : 'which nvidia-smi';
      await execFileAsync(isWindows ? 'cmd' : 'sh', ['-c', checkCmd]);
      hasNv = true;
    } catch {
      hasNv = false;
    }

    if (hasNv) {
      this.gpuBackend = 'nvidia';
      this.startNvLoop();
      return;
    }

    let hasRocm = false;
    try {
      const checkCmd = isWindows ? 'where rocm-smi' : 'which rocm-smi';
      await execFileAsync(isWindows ? 'cmd' : 'sh', ['-c', checkCmd]);
      hasRocm = true;
    } catch {
      hasRocm = false;
    }

    if (hasRocm) {
      this.gpuBackend = 'rocm';
      await this.sampleRocmOneShot();
      return;
    }

    this.gpuBackend = 'none';
    this.latestGpu = {
      hasNvidiaSmi: false,
      hasRocmSmi: false,
      isMac: false,
      gpus: [],
      error: 'Neither nvidia-smi nor rocm-smi found or accessible',
    };
  }

  private async sampleRocmOneShot(): Promise<void> {
    if (this.rocmUnavailable || this.rocmInFlight) return;
    this.rocmInFlight = true;
    try {
      const { stdout } = await execFileAsync('rocm-smi', ROCM_QUERY_ARGS, { env: process.env });
      const gpus = parseRocmOutput(stdout);
      this.latestGpu = { hasNvidiaSmi: false, hasRocmSmi: true, isMac: false, gpus };
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        this.rocmUnavailable = true;
        this.gpuBackend = 'none';
        this.latestGpu = {
          hasNvidiaSmi: false,
          hasRocmSmi: false,
          isMac: false,
          gpus: [],
          error: 'rocm-smi not found or not accessible',
        };
      } else {
        console.error('Monitor: one-shot rocm-smi failed:', err);
      }
    } finally {
      this.rocmInFlight = false;
    }
  }

  // -------------------------------------------------------------------------
  // NVIDIA loop-mode child
  // -------------------------------------------------------------------------
  private startNvLoop(): void {
    if (this.nvUnavailable || this.nvOneShotMode || this.nvChild) return;

    // A hard-killed server (SIGKILL never runs the exit hook) can orphan the
    // resident loop child. Our exact query string only ever appears in
    // children we spawned, so reap any stray once at boot — after it
    // completes, to not race the kill against our own fresh child.
    if (!this.nvReaped && process.platform !== 'win32') {
      this.nvReaped = true;
      // No loop flag in the pattern so strays from any past cadence match
      execFile('pkill', ['-9', '-f', `nvidia-smi ${NV_QUERY_ARGS.join(' ')}`], () => this.startNvLoop());
      return;
    }
    this.nvReaped = true;

    let child: ChildProcess;
    try {
      child = spawn('nvidia-smi', [...NV_QUERY_ARGS, '-lms', String(MONITOR_TICK_MS)], {
        env: NV_ENV,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
    } catch {
      this.markNvUnavailable();
      return;
    }
    this.nvChild = child;
    this.nvStdoutBuffer = '';
    this.nvBatch = [];
    this.lastNvLineAt = Date.now();

    child.stdout!.on('data', (chunk: Buffer) => this.onNvData(chunk.toString()));
    child.stderr!.on('data', () => {
      // nvidia-smi warnings are not actionable here
    });
    child.on('error', (err: NodeJS.ErrnoException) => {
      if (err.code === 'ENOENT') {
        this.markNvUnavailable();
      }
    });
    child.on('exit', () => {
      if (this.nvChild !== child) return;
      this.nvChild = null;
      if (this.nvFlushTimer) {
        clearTimeout(this.nvFlushTimer);
        this.nvFlushTimer = null;
      }
      if (!this.nvUnavailable && !this.nvOneShotMode) {
        setTimeout(() => this.startNvLoop(), 5000);
      }
    });
  }

  private onNvData(text: string): void {
    this.nvStdoutBuffer += text;
    const lines = this.nvStdoutBuffer.split('\n');
    this.nvStdoutBuffer = lines.pop() || '';
    for (const line of lines) {
      if (!line.trim()) continue;
      const gpu = parseGpuLine(line);
      if (!gpu) continue;
      this.nvEverGotLine = true;
      this.lastNvLineAt = Date.now();
      this.nvBatch.push(gpu);
    }
    if (this.nvFlushTimer) clearTimeout(this.nvFlushTimer);
    this.nvFlushTimer = setTimeout(() => this.flushNvBatch(), NV_BATCH_FLUSH_MS);
  }

  private flushNvBatch(): void {
    this.nvFlushTimer = null;
    if (this.nvBatch.length === 0) return;
    this.latestGpu = {
      hasNvidiaSmi: true,
      hasRocmSmi: false,
      isMac: false,
      gpus: this.nvBatch.sort((a, b) => a.index - b.index),
    };
    this.nvBatch = [];
  }

  private nvWatchdog(): void {
    if (this.nvUnavailable || !this.nvChild) return;
    if (Date.now() - this.lastNvLineAt <= NV_WATCHDOG_MS) return;
    console.warn('Monitor: no output from nvidia-smi loop, restarting it');
    // Loop mode that never produced a single line isn't going to start —
    // switch to a one-shot spawn per tick instead of kill/respawn forever.
    if (!this.nvEverGotLine) {
      this.nvOneShotMode = true;
    }
    this.nvChild.kill('SIGKILL');
  }

  private async sampleNvOneShot(): Promise<void> {
    if (this.nvUnavailable || this.nvOneShotInFlight) return;
    this.nvOneShotInFlight = true;
    try {
      const { stdout } = await execFileAsync('nvidia-smi', NV_QUERY_ARGS, { env: NV_ENV });
      const gpus = stdout
        .trim()
        .split('\n')
        .map(parseGpuLine)
        .filter((gpu): gpu is GpuInfo => gpu !== null)
        .sort((a, b) => a.index - b.index);
      this.latestGpu = { hasNvidiaSmi: true, hasRocmSmi: false, isMac: false, gpus };
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        this.markNvUnavailable();
      } else {
        console.error('Monitor: one-shot nvidia-smi failed:', err);
      }
    } finally {
      this.nvOneShotInFlight = false;
    }
  }

  private async markNvUnavailable(): Promise<void> {
    this.nvUnavailable = true;
    if (this.gpuBackend === 'rocm') return;
    try {
      const isWindows = process.platform === 'win32';
      const checkCmd = isWindows ? 'where rocm-smi' : 'which rocm-smi';
      await execFileAsync(isWindows ? 'cmd' : 'sh', ['-c', checkCmd]);
      this.gpuBackend = 'rocm';
      await this.sampleRocmOneShot();
      return;
    } catch {
      // rocm not available
    }
    this.gpuBackend = 'none';
    this.latestGpu = {
      hasNvidiaSmi: false,
      hasRocmSmi: false,
      isMac: false,
      gpus: [],
      error: 'Neither nvidia-smi nor rocm-smi found or accessible',
    };
  }
}

/**
 * Idempotent starter. Guarded on globalThis so dev-mode module reloads never
 * stack a second sampler (and a second resident nvidia-smi) in the same
 * process.
 */
export function startMonitor(): SystemMonitor {
  const g = globalThis as unknown as { __aiToolkitSystemMonitor?: SystemMonitor };
  if (!g.__aiToolkitSystemMonitor) {
    g.__aiToolkitSystemMonitor = new SystemMonitor();
    // Escape hatch: with AI_TOOLKIT_DISABLE_MONITOR=1 the monitor object
    // exists (the SSE route stays functional) but never samples anything.
    if (process.env.AI_TOOLKIT_DISABLE_MONITOR !== '1') {
      g.__aiToolkitSystemMonitor.start();
    }
  }
  return g.__aiToolkitSystemMonitor;
}
