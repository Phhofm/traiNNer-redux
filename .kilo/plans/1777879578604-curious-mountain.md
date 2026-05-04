# MCP Server Integration Analysis for traiNNer-redux

## Executive Summary

Analysis of 5 MCP servers for enhancing Kilo Code's capabilities with a PyTorch-based image super-resolution training codebase. Recommendations prioritize immediate value vs. complexity trade-offs.

---

## Tool Analysis

### 1. Depwire ⭐ **HIGH PRIORITY**

**What it does:** Builds deterministic dependency graphs using tree-sitter across 11 languages, providing exact symbol-level relationships (not probabilistic RAG).

**Why it's critical for traiNNer-redux:**
- Your codebase spans Python (PyTorch training), potentially C++/CUDA ops, config files, scripts — Depwire maps exact symbol dependencies
- "What-if" simulations prevent breaking changes when refactoring architectures, losses, or metrics
- Security scanner is graph-aware (knows what's reachable from attack surfaces)
- Architecture health score tracks coupling, cohesion, circular deps, god files, dead code
- Cross-language edge detection: Python training scripts calling C++ extensions, REST APIs between components

**Direct value for ML training:**
- Before renaming/moving a loss function, see exactly which 47 training scripts, configs, and tests will break
- dead-code detection finds orphaned model architectures, unused loss weights, deprecated schedulers
- Visualize module coupling — identify tightly-coupled components that should be decoupled
- PR impact analysis automates architecture review

**Licensing:** BSL 1.1 (free for personal/internal use) → Apache 2.0 in 2029

**Installation:** `npm install -g depwire-cli` then `depwire mcp` as MCP server

**API key:** ❌ NO — completely local, offline

**Recommendation:** **INTEGRATE** — highest ROI for safe refactoring of complex ML codebase

---

### 2. Open-WebSearch ⭐ **HIGH PRIORITY**

**What it does:** Multi-engine web search + content fetching without API keys. No authentication required.

**Why it's valuable for traiNNer-redux:**
- Research-intensive ML work requires constant lookup: papers, docs, tutorials, GitHub issues
- No API key management — just install and search
- Fetches full article content from CSDN, Juejin (Chinese tech blogs), GitHub READMEs, Linux.do forums
- Multiple engines (Bing, DuckDuckGo, Brave, Baidu) reduce single-point failure
- Can fetch specific article content after searching

**Direct value for ML training:**
- "Latest PyTorch 2.7 best practices for mixed precision training" → get current docs
- "SISR loss function combinations 2024" → research papers, Reddit discussions
- "timm model architecture comparison" → model card details
- Fetch GitHub READMEs of referenced libraries (spandrel, pyvips, etc.)
- Troubleshoot errors: stack traces → search → immediate solutions

**Licensing:** Apache-2.0

**Installation:** `npx open-websearch@latest` or Docker

**API key:** ❌ NO — direct search engine scraping

**Recommendation:** **INTEGRATE** — eliminates context-switching to browser for research

---

### 3. Hindsight ⚠️ **MEDIUM PRIORITY**

**What it does:** Agent memory system that learns over time (not just recall). Retain/recall/reflect paradigm with biomimetic data structures.

**Why it's less critical:**
- Your primary need is codebase understanding + research, not conversational memory
- Kilo Code already has session memory; Hindsight adds long-term learning but at operational cost
- Requires separate Docker container + API server + client library
- More valuable for chatbots/agents with persistent user interactions
- Your use case: engineer working on code, not conversational AI with user history

**Potential use cases (if integrated):**
- Remember successful training configurations (learning rates, batch sizes) for specific dataset characteristics
- Recall previous experiment outcomes and insights
- Track what architectural changes improved PSNR/SSIM metrics

**Licensing:** MIT

**Installation:** Docker container + connect via MCP

**API key requirements:** ✅ YES — BUT can use **FREE options**:
  - **Ollama locally** (no key): `HINDSIGHT_API_LLM_PROVIDER=ollama` + `HINDSIGHT_API_LLM_BASE_URL=http://localhost:11434/v1`
  - **Kilo Gateway** (if OpenAI-compatible): Set `HINDSIGHT_API_LLM_PROVIDER=openai` + `HINDSIGHT_API_LLM_BASE_URL=<kilo-gateway-endpoint>`
  - **Claude Code** (uses your Claude subscription): `HINDSIGHT_API_LLM_PROVIDER=claude-code` (no separate API key, uses `claude auth login`)
  - **Built-in llama.cpp** (auto-downloads 3.5GB GGUF): `HINDSIGHT_API_LLM_PROVIDER=llamacpp` (fully local, no key)

**Recommendation:** **DEFER** — nice-to-have but not essential. Consider after Depwire + Open-WebSearch are working. If you have Ollama already running, it's a zero-cost addition.

---

### 4. Firecrawl ⚠️ **LOW PRIORITY**

**What it does:** Full-featured web scraping API (search, scrape, crawl, map, agent mode). More powerful than Open-WebSearch but requires API key for cloud version.

**Why it's overkill:**
- You already have Open-WebSearch for basic search + fetch
- Firecrawl's advanced features (JS rendering, agent mode, batch scraping) are for data ingestion pipelines, not code assistant context
- Self-hosted Docker version exists but requires PostgreSQL, Redis, Playwright microservice — heavier stack
- AGPL-3.0 license may affect codebase if modified
- Your use case: documentation lookup, research, not large-scale web data collection

**When you'd need it:**
- If you start scraping model zoos, dataset repositories, or paper PDFs at scale
- Building automated dataset curation pipelines from the web

**Licensing:** AGPL-3.0 (open source) with commercial cloud option

**Self-hosted Docker:** YES — works completely without API key. See SELF_HOST.md. Requires:
  - Docker Compose (PostgreSQL, Redis, Playwright, main API)
  - Port 3002 exposed
  - No API keys needed for self-hosted instance

**API key for cloud:** ✅ YES required for cloud version (`api.firecrawl.dev`)

**Recommendation:** **SKIP** — Open-WebSearch covers 90% of need with less complexity. Only add if you need advanced scraping (JS-heavy pages, batch crawling, agent mode).

---

### 5. GitNexus ⚠️ **CONSIDER AFTER DEPWIRE**

**What it does:** Code intelligence platform building a cross-language knowledge graph with execution flow tracing, clustering, and smart tools. Similar to Depwire but more feature-rich.

**Comparison with Depwire:**

| Feature | Depwire | GitNexus |
|---------|---------|----------|
| Core tech | Tree-sitter AST | Tree-sitter + custom resolution + LadybugDB graph |
| Languages | 11 | 15+ |
| Output | Dependency graph, health score, dead code | Knowledge graph + clusters + execution processes |
| Tools | `health`, `query`, `deps`, `pr-impact` | `impact`, `query`, `context`, `detect_changes`, `rename`, `cypher`, `group_*` |
| Multi-repo | No | Yes (monorepo/multi-service) |
| License | BSL 1.1 (free personal) | MIT |
| Indexing | On-demand, fast | Initial scan slower, persistent `~/.gitnexus/` |
| LLM integration | Basic tools | Agent skills + prompts + Claude hooks |

**Why consider GitNexus:**
- Execution flow tracing ("Show full call chain from training loop → loss calculation")
- Built-in rename refactoring that updates all references
- Cypher queries for ad-hoc graph exploration
- Agent skills auto-generated from codebase clusters
- Web UI for visual graph browsing (nice for demos/team sync)

**Why defer:**
- Depwire already covers your core need: dependency mapping + impact analysis
- GitNexus adds persistent index (~500MB-2GB), requires `gitnexus analyze` after changes
- Depwire simpler: no separate index step, results from current parse
- Overlap is significant; GitNexus is more complex

**Recommendation:** **PHASE 2** — start with Depwire. Re-evaluate after 2-4 weeks:
- If you need execution flow tracing or multi-repo support → add GitNexus
- If Depwire's impact analysis feels shallow → GitNexus provides richer pre-structured context

GitNexus MCP (for future reference):
```json
{
  "mcp": {
    "gitnexus": {
      "type": "local",
      "command": ["npx", "-y", "gitnexus@latest", "mcp"],
      "enabled": true
    }
  }
}
```

---

### 5. GitNexus ⚠️ **CONSIDER AFTER DEPWIRE**

**What it does:** Code intelligence platform that builds a cross-language knowledge graph with execution flow tracing, clustering, and smart tools. Similar to Depwire but more feature-rich (impact analysis, process tracing, rename refactoring, Cypher queries).

**Comparison with Depwire:**

| Feature | Depwire | GitNexus |
|---------|---------|----------|
| Core tech | Tree-sitter AST parsing | Tree-sitter + custom resolution + LadybugDB graph |
| Languages | 11 (Python, JS/TS, Java, C++, C#, Go, Rust, PHP, Ruby, Kotlin, Swift) | 15+ (same + more framework detection) |
| Output | Dependency graph, health score, dead code | Knowledge graph + clusters + execution processes |
| Tools | `health`, `query`, `deps`, `pr-impact` | `impact`, `query`, `context`, `detect_changes`, `rename`, `cypher`, `group_*` |
| Multi-repo | No | Yes (monorepo/multi-service support) |
| License | BSL 1.1 (free personal) | MIT (more permissive) |
| Indexing | On-demand, fast | Initial scan slower, persistent index in `.gitnexus/` |
| LLM integration | Basic tools | Built-in agent skills + prompts + Claude Code hooks |

**Why you might want GitNexus:**
- Execution flow tracing ("Show me the full call chain from training loop to loss calculation")
- Cross-repo analysis if you ever split traiNNer-redux into multiple packages
- Built-in rename refactoring that updates all references safely
- Cypher query language for ad-hoc graph queries
- More automated agent skills (Exploring, Debugging, Impact Analysis, Refactoring)
- Web UI for visual graph exploration (nice for demos)

**Why you might skip it:**
- Depwire already covers the core need: **dependency mapping, impact analysis, health scoring**
- GitNexus requires persistent indexing (~500MB-2GB storage for large repos) and re-indexing on changes
- Additional setup step (`gitnexus analyze` after every structural change)
- Depwire gives you 80% of the value with simpler workflow
- You already have Hindsight for memory, Open-WebSearch for research — GitNexus competes more directly with Depwire

**Recommendation:** **DEFER** — start with Depwire. Evaluate after 2-4 weeks of use:
- Are you hitting Depwire's limitations?
- Do you need execution flow tracing or multi-repo support?
- Is the impact analysis not detailed enough?

If yes, add GitNexus as a **complement** (not replacement) — they have different strengths. Depwire is lighter, faster, focused on architectural health. GitNexus is deeper, more structured, better for code exploration.

---

## Recommended Integration Order

### Phase 1 (Immediate — 4 Tools)
All four tools provide distinct value and work well together:

1. **Depwire** — codebase intelligence, prevents breaking changes
2. **Open-WebSearch** — research lookup without browser switching
3. **Hindsight** — persistent memory for experiments, configurations, insights
4. **Firecrawl** — advanced web scraping when needed (JS-heavy pages, batch operations)

**Rationale for all-at-once:**
- Depwire + Open-WebSearch: zero API keys, immediate ROI
- Hindsight: uses your existing Ollama container (free, local LLM)
- Firecrawl: self-hosted Docker (no API key), complements Open-WebSearch for tough scraping jobs
- All four together create a complete vibe coding environment: understands code, remembers history, researches current info, scrapes stubborn sites

**Total setup time:** ~20-30 minutes (most already running).

### Phase 2 (Optional — GitNexus)
After using Depwire for 2-4 weeks, evaluate if you need deeper code intelligence:

**Consider GitNexus if:**
- You need execution flow tracing (from training entry point → loss → backward)
- You want built-in rename refactoring with full confidence
- You work with multiple repos/monorepos and need cross-repo impact analysis
- You want visual graph exploration for team demos

**Skip if:**
- Depwire's dependency mapping + health scores already meet your needs
- Simpler workflow (no separate indexing step) is preferred

GitNexus setup: `npm install -g gitnexus && gitnexus analyze` then add MCP entry as shown above.

---

## Configuration Examples

### Depwire MCP Setup

Add to global Kilo Code config at `~/.config/kilo/kilo.json` (create directory/file if missing):

```json
{
  "mcp": {
    "depwire": {
      "type": "local",
      "command": ["npx", "-y", "depwire-cli", "mcp"],
      "enabled": true
    }
  }
}
```

**First run:** Depwire will parse the entire codebase (~few seconds for traiNNer-redux size). After that, tool calls are instant.

**Test:** Ask Kilo Code "What does Depwire show for the architecture health of this project?"

**First run:** Depwire will parse the entire codebase (~few seconds for traiNNer-redux size). After that, tool calls are instant.

**Test:** Ask Kilo Code "What does Depwire show for the architecture health of this project?"

---

### Open-WebSearch MCP Setup

**Preferred: Docker daemon mode (avoid Node version issues)**

Container already running on port 3000. Add to `~/.config/kilo/kilo.json`:

```json
{
  "mcp": {
    "web-search": {
      "type": "remote",
      "url": "http://localhost:3000/mcp",
      "enabled": true,
      "timeout": 30000
    }
  }
}
```

**Alternative (stdio via npx):** If you have Node ≥20.18.1, use local mode:
```json
{
  "mcp": {
    "web-search": {
      "type": "local",
      "command": ["npx", "-y", "open-websearch@latest"],
      "environment": {
        "MODE": "stdio",
        "DEFAULT_SEARCH_ENGINE": "duckduckgo",
        "ALLOWED_SEARCH_ENGINES": "duckduckgo,bing,exa,brave"
      },
      "enabled": true
    }
  }
}
```

**Test:** "Search for latest PyTorch 2.7 mixed precision training best practices"

---

### Hindsight MCP Setup (Using Ollama — No API Key)

**Status:** Hindsight API already running on port 8888 via Python venv at `.venv-hindsight`. Ollama running with qwen2.5:7b.

**Prerequisite:** Verify Hindsight API is accessible:
```bash
curl http://localhost:8888/health  # should return {"status":"ok"}
```

**MCP Configuration:** Install Hindsight client in venv if needed:
```bash
.venv-hindsight/bin/pip install hindsight-client
```

Then add to `~/.config/kilo/kilo.json`:
```json
{
  "mcp": {
    "hindsight": {
      "type": "remote",
      "url": "http://localhost:8888/mcp",
      "enabled": true,
      "timeout": 10000
    }
  }
}
```
- If MCP server insists on API key, you can set a dummy value.

**Test:** "Search the web for recent SISR papers and scrape the first result"

---

## Combined MCP Configuration Example

Global Kilo Code config at `~/.config/kilo/kilo.json`:

```json
{
  "mcp": {
    "depwire": {
      "type": "local",
      "command": ["npx", "-y", "depwire-cli", "mcp"],
      "enabled": true
    },
    "web-search": {
      "type": "remote",
      "url": "http://localhost:3000/mcp",
      "enabled": true,
      "timeout": 30000
    },
    "hindsight": {
      "type": "remote",
      "url": "http://localhost:8888/mcp",
      "enabled": true,
      "timeout": 10000
    },
    "firecrawl": {
      "type": "local",
      "command": ["npx", "-y", "firecrawl-mcp"],
      "environment": {
        "FIRECRAWL_API_URL": "http://localhost:3002"
      },
      "enabled": true,
      "timeout": 60000
    }
  }
}
```

**Note:** Hindsight MCP is served directly by the Hindsight API at `/mcp`. Depwire and Firecrawl use local stdio via npx.

---

## Implementation Checklist

### Prerequisites
- [x] Node.js ≥18 installed (for depwire, firecrawl-mcp; open-websearch uses Docker)
- [x] Docker + Docker Compose installed (Firecrawl running)
- [x] Ollama running locally with qwen2.5:7b model
- [x] Kilo Code global MCP config location identified: `~/.config/kilo/kilo.json`

### ✅ Already Completed

**Depwire:**
- [x] `npm install -g depwire-cli` (v1.0.8 installed)
- [x] Initial parse run, health score available
- [ ] Add MCP config entry to `~/.config/kilo/kilo.json`
- [ ] Restart Kilo Code
- [ ] Test: "Show me the dependency graph of traiNNer-redux"

**Open-WebSearch:**
- [x] Docker container running on port 3000
- [ ] Add MCP config entry to `~/.config/kilo/kilo.json` (remote HTTP type)
- [ ] Restart Kilo Code
- [ ] Test: "Search for PyTorch DataLoader best practices"

**Hindsight:**
- [x] Python venv created at `/home/phhofm/Documents/GitHub/traiNNer-redux/.venv-hindsight`
- [x] Hindsight API running on port 8888 with Ollama qwen2.5:7b
- [ ] Install `hindsight-client` in venv: `./.venv-hindsight/bin/pip install hindsight-client`
- [ ] Add MCP config entry to `~/.config/kilo/kilo.json` (use absolute path: `/home/phhofm/Documents/GitHub/traiNNer-redux/.venv-hindsight/bin/python`)
- [ ] Restart Kilo Code
- [ ] Test: "Retain this: 'traiNNer-redux uses EMA for model averaging'"

**Firecrawl:**
- [x] Docker stack running (PostgreSQL, Redis, RabbitMQ, Playwright, API on 3002)
- [x] API verified working: POST `/v1/crawl` returns job ID
- [ ] Install `firecrawl-mcp`: `npm install -g firecrawl-mcp`
- [ ] Add MCP config entry to `~/.config/kilo/kilo.json`
- [ ] Restart Kilo Code
- [ ] Test: "Crawl the traiNNer-redux GitHub repo README"

### Remaining Execution Steps

**Step 1: Ensure config directory exists**
```bash
mkdir -p ~/.config/kilo
```

**Step 2: Create/overwrite `~/.config/kilo/kilo.json` with this content:**

```json
{
  "mcp": {
    "depwire": {
      "type": "local",
      "command": ["npx", "-y", "depwire-cli", "mcp"],
      "enabled": true
    },
    "web-search": {
      "type": "remote",
      "url": "http://localhost:3000/mcp",
      "enabled": true,
      "timeout": 30000
    },
    "hindsight": {
      "type": "remote",
      "url": "http://localhost:8888/mcp",
      "enabled": true,
      "timeout": 10000
    },
    "firecrawl": {
      "type": "local",
      "command": ["npx", "-y", "firecrawl-mcp"],
      "environment": {
        "FIRECRAWL_API_URL": "http://localhost:3002"
      },
      "enabled": true,
      "timeout": 60000
    }
  }
}
```

**Step 3: Install missing dependency**
```bash
npm install -g firecrawl-mcp
# hindsight-client not needed — Hindsight API already has MCP built-in
```

**Step 4: Restart Kilo Code**
- Close and reopen VS Code/Kilo Code window
- Or run `kilo restart` if using CLI

**Step 5: Verify**
- Open Settings → Agent Behaviour → MCP Servers
- All 4 servers should show green "Connected" status
- If any show "Failed", check Output panel for errors

---

## Risks & Considerations

- **Depwire BSL license:** Free for personal/internal use; converts to Apache 2.0 in 2029. Commercial use requires license.
- **Open-WebSearch rate limits:** Excessive search may trigger engine blocks. Use judiciously.
- **Resource overhead:** Depwire parses entire codebase initially (~few seconds for your size). Open-WebSearch runs lightweight Node.js daemon.
 - **No MCP config found:** You'll be starting fresh — good, no conflicts.

---

## Critical Configuration Note

### ⚠️ MCP Config Location in Kilo Code

Kilo Code does **NOT** use VS Code's `~/.config/Code/User/mcp.json`. It uses its own config:

**Use global config** (`~/.config/kilo/kilo.json`) for these infrastructure tools — they'll be available in every project.

After creating/editing the config:
1. Save `~/.config/kilo/kilo.json`
2. **Restart Kilo Code** (VS Code window reload) to detect new servers
3. Open **Settings → Agent Behaviour → MCP Servers**
4. All 4 servers should appear with a green "Connected" status
5. If any show "Failed", check the Kilo Code output panel: View → Output → "Kilo Code"

### Verify Commands

Quick checks to confirm each server/API is reachable:
```bash
# Depwire (local stdio — try inside Kilo Code after restart)
depwire health /home/phhofm/Documents/GitHub/traiNNer-redux

# Open-WebSearch (remote HTTP endpoint)
curl -s http://localhost:3000/mcp | head -20

# Hindsight API (separate from MCP wrapper)
curl http://localhost:8888/health  # {"status":"ok"}

# Firecrawl API (separate from MCP wrapper)
curl http://localhost:3002/v1/health  # {"status":"healthy"}
```

If a server shows "Failed" in Kilo Code:
- Ensure the background process/container is running
- Check command/path correctness in config (absolute paths recommended)
- Verify environment variables (FIRECRAWL_BASE_URL)
- Check Kilo Code output for specific error messages

### GitNexus Note

GitNexus is **NOT** included in the initial integration because:
- Depwire already covers core dependency analysis needs
- GitNexus uses a registry-based architecture (`~/.gitnexus/registry.json`) rather than per-project config
- Would add 500MB-2GB indexing overhead and re-indexing maintenance
- Consider as **Phase 2** if Depwire proves insufficient for execution flow tracing needs

---

## Expected Benefits

**With all four tools integrated, Kilo Code becomes:**

1. **Architecture-aware** (Depwire) — knows exact symbol dependencies, can simulate refactors before breaking code, detects dead code, scores architecture health
2. **Research-connected** (Open-WebSearch) — instant lookup of docs, papers, GitHub issues without browser switching
3. **Experiment-aware** (Hindsight + Ollama) — remembers which training configs worked, tracks insights across sessions, builds mental models of your codebase
4. **Scraping-capable** (Firecrawl) — handles JS-heavy documentation sites, batch crawls model repos, extracts structured data from any page

**Quantified benefits:**
- Depwire: 40% fewer tool calls, 56% fewer file reads (token-efficient surgical reads)
- Open-WebSearch: ~5 seconds per research query vs ~30 seconds switching to browser
- Hindsight: cumulative knowledge across months of experiments (no more "what LR did I use last month?")
- Firecrawl: access to content other scrapers can't reach (SPA/JS-heavy sites)

**Combined effect:** Kilo Code transitions from assistant to pair programmer that understands your codebase, remembers your history, and can fetch current information on demand.

---

## Risks & Considerations

| Tool | Risk | Mitigation |
|------|------|------------|
| Depwire (BSL 1.1) | License converts to Apache 2.0 in 2029; commercial use requires license | Personal/internal use is free; you're an individual developer — compliant |
| Open-WebSearch | Rate limits from search engines | Use multiple engines (DuckDuckGo, Brave, Bing), add delays, cache frequent queries |
| Hindsight (Ollama) | LLM quality depends on local model | Use qwen2.5:7b or larger; switch to cloud LLM if needed |
| Firecrawl (Docker) | Heavier resource usage (~2GB RAM for full stack) | Stack already running; monitor resource usage |

**Total resource overhead:** ~3-4GB RAM when all running, ~1GB idle. Negligible on dev machines with 16GB+.

**Note:** Starting with fresh Kilo Code MCP config — no conflicts with existing servers expected.

---

## 📦 Replication Guide: Universal Kilo Code MCP Setup

### Concept

This setup creates a **global MCP server configuration** in `~/.config/kilo/kilo.json` that provides AI-powered development tools available in **every project** you open with Kilo Code. It's a "dev environment as code" approach — once set up on a machine, all your projects instantly gain:

- **Codebase intelligence** (Depwire) — dependency graphs, health scores, impact analysis
- **Live research** (Open-WebSearch) — multi-engine web search without leaving editor
- **Persistent memory** (Hindsight) — AI remembers experiments, configs, insights across sessions
- **Advanced scraping** (Firecrawl) — JS-heavy sites, batch crawling, structured extraction
- **Documentation lookup** (Context7) — instant library/docs search
- **Structured reasoning** (Sequential Thinking) — step-by-step problem decomposition
- **Browser automation** (Playwright) — AI can test web UIs, fill forms, take screenshots

### System Requirements

| Component | Requirement | Why |
|-----------|-------------|-----|
| **OS** | Linux/macOS/Windows | Cross-platform |
| **Node.js** | v18+ (v20+ recommended) | Runs Depwire, Firecrawl-MCP, Context7, SequentialThinking |
| **Docker + Compose** | Latest | Runs Firecrawl stack (PostgreSQL, Redis, RabbitMQ, Playwright service) |
| **Ollama** | Latest | Local LLM for Hindsight memory (no API key) |
| **Python 3.10+** | With pip | Hindsight API server |
| **RAM** | 8GB+ (16GB recommended) | All services: ~3-4GB; Firecrawl stack ~2GB alone |
| **Disk** | 10GB free | Docker images, indexes, caches |

### One-Time Setup (30-45 min)

#### 1. Install Base Tools
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y docker.io docker-compose python3.10-venv python3-pip nodejs npm

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# macOS (with Homebrew)
brew install docker docker-compose python3 node@20 ollama

# Windows (with Winget)
winget install Docker.DockerDesktop
winget install Python.Python.3.10
winget install Nodejs.20
winget install Ollama
```

#### 2. Install & Start Ollama
```bash
# Start Ollama service
ollama serve &

# Pull a model (choose one)
ollama pull qwen2.5:7b    # 4.7GB, fast
ollama pull llama3:8b    # 4.7GB, balanced
ollama pull mistral      # 4.1GB, good for coding
```

#### 3. Start Firecrawl Stack (Docker)
```bash
git clone https://github.com/firecrawl/firecrawl.git /tmp/firecrawl
cd /tmp/firecrawl
cp .env.example .env
# Edit .env: set PORT=3002, generate BULL_AUTH_KEY
docker compose up -d

# Verify
curl http://localhost:3002/v1/health  # {"status":"healthy"}
```

#### 4. Install NPM Global Packages
```bash
npm install -g depwire-cli
npm install -g firecrawl-mcp
# Context7 and SequentialThinking will be fetched on-demand by npx
```

#### 5. Start Hindsight API (Python venv)
```bash
# Create venv (once)
cd /path/to/your/workdir  # e.g., ~/Documents/GitHub/traiNNer-redux
python3 -m venv .venv-hindsight

# Install Hindsight with local LLM support
.venv-hindsight/bin/pip install "hindsight-all[local-llm]"

# Start Hindsight API with Ollama backend
export HINDSIGHT_API_LLM_PROVIDER=ollama
export HINDSIGHT_API_LLM_BASE_URL=http://localhost:11434/v1
export HINDSIGHT_API_LLM_MODEL=qwen2.5:7b
export HINDSIGHT_API_DATABASE_URL=pg0://hindsight-mcp  # embedded SQLite via pg0
.venv-hindsight/bin/hindsight-api &
# Or use: .venv-hindsight/bin/hindsight-local-mcp &

# Verify
curl http://localhost:8888/health  # {"status":"ok"}
```

#### 6. Start Open-WebSearch (Docker)
```bash
docker run -d -p 3000:3000 ghcr.io/aas-ee/open-web-search:latest

# Verify
curl -s http://localhost:3000/mcp | head -5  # should return {}
```

#### 7. Configure Kilo Code Global MCP
Create `~/.config/kilo/kilo.json`:

```json
{
  "mcp": {
    "depwire": {
      "type": "local",
      "command": ["npx", "-y", "depwire-cli", "mcp"],
      "enabled": true
    },
    "web-search": {
      "type": "remote",
      "url": "http://localhost:3000/mcp",
      "enabled": true,
      "timeout": 30000
    },
    "hindsight": {
      "type": "remote",
      "url": "http://localhost:8888/mcp",
      "enabled": true,
      "timeout": 10000
    },
    "firecrawl": {
      "type": "local",
      "command": ["npx", "-y", "firecrawl-mcp"],
      "environment": {
        "FIRECRAWL_API_URL": "http://localhost:3002"
      },
      "enabled": true,
      "timeout": 60000
    },
    "context7": {
      "type": "local",
      "command": ["npx", "-y", "@upstash/context7-mcp"],
      "environment": {
        "DEFAULT_MINIMUM_TOKENS": "{{DEFAULT_MINIMUM_TOKENS}}"
      },
      "enabled": true
    },
    "sequentialthinking": {
      "type": "local",
      "command": ["npx", "-y", "@modelcontextprotocol/server-sequential-thinking"],
      "enabled": true
    },
    "playwright": {
      "type": "local",
      "command": ["npx", "-y", "@playwright/mcp"],
      "enabled": true
    }
  }
}
```

**Notes:**
- Use `"type": "remote"` for HTTP-based servers (Open-WebSearch, Hindsight)
- Use `"type": "local"` for stdio-based servers (everything else)
- Adjust timeouts if needed (default: 10s local, 30s remote)
- Context7 token minimum can be adjusted or removed

#### 8. Restart Kilo Code
- Close and reopen VS Code/Kilo Code window
- Open Settings → Agent Behaviour → MCP Servers
- All servers should show green "Connected"

### Per-Server Verification

```bash
# Depwire — test dependency analysis
depwire health /path/to/your/project

# Open-WebSearch — test search endpoint
curl -s http://localhost:3000/mcp -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'

# Hindsight — test MCP endpoint
curl -s http://localhost:8888/mcp -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'

# Firecrawl — test API (separate from MCP)
curl -s -X POST http://localhost:3002/v1/crawl \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com","limit":1}'
```

### What Each Server Provides

| Server | Purpose | Tools | When to use |
|--------|---------|-------|-------------|
| **Depwire** | Code dependency analysis | `health`, `deps`, `query`, `pr-impact` | Before refactoring; find dead code; understand architecture |
| **Open-WebSearch** | Web search | `search`, `fetch` | Research papers, docs, StackOverflow, GitHub issues |
| **Hindsight** | Persistent memory | `retain`, `recall`, `reflect` | Remember experiment configs, insights, past decisions |
| **Firecrawl** | Advanced scraping | `scrape`, `crawl`, `map`, `search` | JS-heavy sites, batch crawling, structured extraction |
| **Context7** | Library documentation | `get_docs`, `search_symbols` | Instant API lookup for popular libraries |
| **SequentialThinking** | Structured reasoning | `think` (multi-step) | Complex problem decomposition, planning |
| **Playwright** | Browser automation | `navigate`, `click`, `fill`, `screenshot` | Web app testing, interactive scraping, UI verification |

### Copying to Another Machine

1. **Export config:** Copy `~/.config/kilo/kilo.json` to new machine's same location
2. **Install prerequisites:** Node.js, Docker, Ollama, Python (see step 1)
3. **Start services:** Run steps 2-6 (start Docker containers + Hindsight API)
4. **Install NPM packages:** `npm install -g depwire-cli firecrawl-mcp`
5. **Restart Kilo Code:** Done — all tools available

### Resource Management

**Start all services:**
```bash
# Firecrawl stack
docker compose -f /tmp/firecrawl/docker-compose.yaml up -d

# Hindsight API
cd ~/Documents/GitHub/traiNNer-redux && .venv-hindsight/bin/hindsight-api &

# Verify all ports
lsof -i :3000  # Open-WebSearch
lsof -i :3002  # Firecrawl
lsof -i :8888  # Hindsight
```

**Stop all services:**
```bash
docker compose -f /tmp/firecrawl/docker-compose.yaml down
pkill -f "hindsight-api"
pkill -f "ollama"
```

**Auto-start on boot (Linux systemd):** Create service files for Docker stacks and Hindsight/Ollama.

### Troubleshooting

| Issue | Fix |
|-------|-----|
| Server shows "Failed" in Kilo Code | Check output: View → Output → "Kilo Code"; verify process listening on expected port |
| Depwire slow first run | Normal — it parses entire codebase on first query (~10-30s for large repos) |
| Open-WebSearch returns errors | Search engines may rate-limit; try different engines or add delays |
| Hindsight connection refused | Ensure `.venv-hindsight/bin/hindsight-api` running; check Ollama is up |
| Firecrawl MCP env error | Use `FIRECRAWL_API_URL` not `FIRECRAWL_BASE_URL`; Firecrawl container must be healthy |
| Context7 slow | First run downloads model; subsequent calls faster |
| Playwright browsers not found | Run `npx @playwright/mcp` once to auto-install browsers, or manually: `npx playwright install chromium` |

### Customization

- **Switch LLM for Hindsight:** Change `HINDSIGHT_API_LLM_PROVIDER` to `openai`, `anthropic`, or `claude-code` (uses your Claude subscription)
- **Switch search engines:** Set `DEFAULT_SEARCH_ENGINE=bing` in Open-WebSearch Docker env
- **Add GitNexus later:** `npm install -g gitnexus && gitnexus analyze`, then add MCP entry
- **Per-project overrides:** Create `.kilo/kilo.json` in project root to override global servers

---

## Final Recommendation

This global MCP setup transforms Kilo Code into a **full-stack AI development environment** — it understands your code, remembers your history, fetches current information, and can automate browser tasks. Once configured on a machine, it works in **every project** automatically.

**Maintenance:**
- Keep Docker containers updated: `docker compose pull && docker compose up -d`
- Update NPM packages periodically: `npm update -g depwire-cli firecrawl-mcp`
- Hindsight indexes auto-refresh; no manual intervention needed
- Ollama models update: `ollama pull qwen2.5:7b` (pulls latest)

**Total cost:** $0 (all tools use local/free resources)
**Time to value:** ~1 hour setup, then immediate productivity gains on every project.
