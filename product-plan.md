# Agent Host Product Plan

## 1. Product vision

Build the easiest place for individuals and small teams to start, watch, pause, resume, and collaborate with AI agents running on either their own computer or cloud compute.

The product should feel like one dependable computer for every agent, even though work may actually run through local Docker, Modal, Colab, AWS, Azure, or another provider.

**Core promise:** Run AI agents without understanding infrastructure and without wondering what they are doing.

**Positioning:** The easiest way to run every agent, on any computer, without babysitting it. Leave safely and come back to proof.

## 2. Initial target customer

The initial audience is not a large enterprise infrastructure team. It is the growing middle of people already trying to use agents but struggling with terminals, cloud setup, disconnected runtimes, hidden progress, and lost work:

- Students and researchers
- Creators and technical artists
- Solo founders
- Freelancers and independent developers
- Small technical teams
- Less-technical users who need agents to complete real computer work

The interface should hide containers, machine types, cloud regions, tokens, and GPU configuration by default. These controls can remain available in an advanced view.

## 3. Problem

Current agent infrastructure is fragmented:

- Local agents stop when the computer sleeps or loses power.
- Cloud runtimes are difficult for average users to configure.
- Long tasks frequently lose progress when a runtime disconnects.
- Users cannot easily tell whether an agent is working, blocked, wasting time, or finished.
- Different agents and compute providers require separate setup and monitoring.
- Collaboration between people and agents is poorly represented.
- Agents often claim completion without presenting trustworthy evidence.
- Users cannot easily move work between a laptop, cloud CPU, browser sandbox, and GPU.

## 4. Competitive landscape

### Modal

**Advantage:** Direct programmable serverless GPU and CPU execution. Modal is strong for inference, fine-tuning, training, batch workloads, autoscaling endpoints, and high-performance GPU access without Kubernetes administration.

**What to adopt:** Simple compute selection, per-second execution, detached jobs, automatic scaling, persistent volumes, checkpoints, and broad GPU availability.

**Opening:** Modal is powerful infrastructure, but it is not a complete consumer-friendly command center for coordinating agents, understanding their actions, collaborating with other people, and returning to verified completed work.

### Daytona

**Advantage:** Broad programmable computers for agents, including containers, persistent Linux VMs, Windows VMs, GPU environments, snapshots, forks, pause/resume, VNC, and SSH.

**What to adopt:** Full computer lifecycle, persistent environments, snapshots, forks, operating-system flexibility, and secret proxying.

**Opening:** Daytona provides the computer primitive. Users still need a simple supervision, orchestration, collaboration, and verification layer.

### E2B

**Advantage:** A simple sandbox API, fast Firecracker environments, code-interpreter tooling, graphical desktop sandboxes, templates, and an approachable developer integration.

**What to adopt:** Very small SDK surface, fast startup, browser/computer-use support, and reusable environment templates.

**Opening:** It is primarily an embeddable sandbox service rather than the persistent home where consumers manage agents, projects, teammates, proof, and heterogeneous compute.

### Runloop

**Advantage:** Agent evaluation and enterprise trust. Its strongest ideas include reusable scenarios, automated scorers, benchmark suites, traces, credential isolation, and tool-level MCP permissions.

**What to adopt:** Verification scenarios, evaluator scores, audit trails, limited capabilities, and task-bound credentials.

**Opening:** It is optimized for developers and enterprise evaluation rather than an understandable everyday workspace for non-expert users.

### Docker Sandboxes

**Advantage:** Local distribution, familiarity, privacy, persistent microVMs, isolated Docker engines, and support for multiple popular coding agents.

**What to adopt:** Local-first execution, project compatibility, microVM isolation, reusable kits, and a central MCP gateway.

**Opening:** The user's computer still supplies the storage and compute. Work may stop when the laptop closes, local resources are limited, and there is no automatic continuation onto cloud infrastructure.

## 5. Product differentiation

The company should not initially build a proprietary hypervisor or GPU cloud. It should own the control, continuity, trust, and user-experience layer above existing providers.

The durable advantages should become:

1. A provider-neutral project and checkpoint format
2. A unified event history across every agent and runtime
3. Intelligent routing based on cost, hardware, privacy, and duration
4. Secure permissions and approval policies shared across providers
5. Verification records proving what an agent accomplished
6. Collaboration data describing how people and agents complete work together
7. A marketplace of reusable agent workflows and environment templates
8. A recognizable visual experience for supervising autonomous work

Supporting MCP alone is not a moat. The moat is the persistent system behind the MCP connection.

## 6. Core product experience

The default workflow should require almost no infrastructure knowledge:

1. Connect GitHub, open a local folder, or upload a project.
2. Describe the desired outcome in normal language.
3. Select an agent or let the product choose one.
4. Choose **Use my computer**, **Run in the cloud**, **Use a GPU**, or **Choose automatically**.
5. Press **Start**.
6. Watch a plain-English play-by-play.
7. Pause, resume, redirect, approve, retry, or cancel the work.
8. Collaborate with other people and agents in the same project.
9. Receive the output and evidence that it works.

## 7. Unified compute layer

The product should present one compute selector while routing jobs among specialized backends.

| Workload | Preferred execution option |
| --- | --- |
| Private or inexpensive repository work | Local Docker sandbox |
| Fast temporary code execution | E2B-style cloud sandbox |
| Browser and graphical computer use | Desktop sandbox |
| Persistent Linux or Windows environment | Daytona-style VM |
| GPU inference, training, video, or batch work | Modal or another GPU provider |
| Large agent evaluation suite | Runloop-style evaluation infrastructure |
| Customer-controlled enterprise workload | AWS, Azure, GCP, or BYOC |

Routing should consider:

- Required CPU, memory, disk, GPU, and operating system
- Privacy and data-location requirements
- Expected job duration
- Startup latency
- Provider availability
- User budget
- Existing project state and data location
- Whether the environment must remain interactive or persistent

The user may override the choice, but **Choose automatically** should be the recommended default.

## 8. Live supervision

Every active task should expose two synchronized views.

### Simple view

- Working, waiting, blocked, failed, paused, or finished
- Current action in normal English
- Overall progress and current step
- Elapsed time and estimated completion time
- Current and projected cost
- Last successful checkpoint
- Required user decision
- Which agent, person, and runtime own the task

Example updates:

- Reading the repository and identifying the affected files
- Installing the project's dependencies
- Training epoch 8 of 20
- Running 17 of 25 evaluations
- Tests failed; examining the error
- Waiting for approval to deploy
- Checkpoint saved two minutes ago

### Technical view

- Terminal and structured logs
- File diffs
- Tool invocations
- Resource utilization
- Network and permission events
- Checkpoint history
- Model, prompt, agent, and environment versions
- Raw evaluator results

The simple view must summarize real events from the technical view. It must never display invented progress.

## 9. Controls and continuity

Every mission should support:

- Start
- Pause
- Resume
- Redirect
- Retry from the latest checkpoint
- Fork into a separate approach
- Cancel
- Archive
- Transfer to another person or agent
- Move from local execution to cloud execution

Checkpoints should be created at meaningful boundaries and on a time interval during long-running work. The application should store enough provider-neutral state to recover even when the original runtime is gone.

The cloud supervisor should continue monitoring work after the user's laptop closes. If a runtime fails, it should explain the failure and either resume automatically within policy or request a decision.

## 10. Collaboration

A project should act like a shared room for humans and agents:

- Assign tasks to a person, one agent, or several agents
- Display responsibility and current status clearly
- Let teammates comment on a task or artifact
- Hand a task from one agent to another with its context and evidence
- Create isolated branches or workspaces to prevent overwrites
- Compare competing agent outputs
- Require review before merging or publishing
- Maintain a project timeline and decision history
- Notify users only for meaningful blockers, approvals, failures, and completion

## 11. Visual direction

The visual layer can use energetic anime, endless-runner, city, travel, or golf scenes, but every animation must correspond to real task state.

| Visual event | Actual state |
| --- | --- |
| Character running through a city | Agent actively executing steps |
| Character being pursued | Task approaching a budget, deadline, or error limit |
| Character reaching a checkpoint | Durable state successfully saved |
| Character waiting at a door | Human approval required |
| Character playing golf calmly | Healthy long-running background job |
| Character changing vehicles | Job moved to another runtime |
| Team meeting in one location | Agents merging or coordinating work |
| Character celebrating | Tests and evaluator passed |
| Character stopped or injured | Task failed or became blocked |

The animation is an optional supervision mode, not a substitute for the professional dashboard.

## 12. MCP connectivity

The platform should expose a remote MCP server so compatible AI products can directly control the agent host.

### Client-to-host MCP tools

- `list_runtimes`
- `create_workspace`
- `start_agent`
- `select_compute`
- `get_status`
- `get_progress`
- `send_instruction`
- `pause_run`
- `resume_run`
- `cancel_run`
- `approve_action`
- `get_outputs`
- `download_artifact`

This enables requests such as:

> Open this GitHub repository, run two coding agents locally, send the GPU work to Modal, and notify me when every test passes.

The internal host API must be built first. MCP should be a thin adapter over that API so the product is not dependent on one protocol. A REST API, TypeScript SDK, and Python SDK should also be available.

MCP should handle commands, resources, and concise status retrieval. The application's detailed live stream should use WebSockets or server-sent events.

### Agent-to-tool MCP gateway

Agents running inside sandboxes should connect to a separate controlled MCP gateway for services such as GitHub, Google Drive, Slack, databases, and deployment platforms.

The two roles must remain distinct:

- The external AI uses the host MCP server to control missions.
- Sandboxed agents use the tool gateway to access approved external capabilities.

## 13. Security and user control

- Raw credentials should never be placed directly in an agent sandbox.
- Use temporary, mission-bound tokens and server-side credential injection.
- Deny capabilities by default and grant the minimum required tools.
- Make restricted tools invisible to the agent when practical.
- Require confirmation for destructive writes, purchases, deployments, messages, credential changes, and Git pushes.
- Support monetary and resource budgets per mission.
- Apply network and filesystem policies to each sandbox.
- Log every important external action and approval.
- Clearly distinguish proposed actions from completed actions.
- Provide an emergency stop that revokes runtime and tool access.

Avoid exposing an unrestricted remote shell as the primary MCP interface. Higher-level, capability-scoped tools are safer and easier to understand.

## 14. Proof of completion

An agent cannot mark a task complete solely because it says it is finished. Completion should produce a receipt containing the applicable evidence:

- Files changed
- Git diff or generated artifact
- Tests and commands executed
- Test results
- Screenshots or previews
- Evaluator or benchmark score
- Logs and relevant errors
- Model, prompt, agent, and environment versions
- Runtime, duration, and cost
- Tool and permission usage
- Human approvals
- Known limitations or unverified claims

Users should be able to open the receipt, reproduce the run where possible, and share it with a collaborator.

## 15. Recommended MVP

The first release should remain narrow enough to build and validate quickly.

### MVP integrations

1. Local Docker execution
2. One cloud CPU/GPU provider, preferably Modal
3. GitHub repository connection
4. One or two supported coding agents
5. Provider-neutral mission and checkpoint records
6. Plain-English live timeline plus technical logs
7. Start, pause, resume, cancel, and retry
8. Secure credential broker and approval gates
9. Completion receipts with diffs and test evidence
10. Basic project sharing and task assignment
11. Remote MCP control interface

### Explicitly defer

- Building a proprietary GPU fleet
- Supporting every cloud provider at launch
- Universal live migration of an in-memory VM
- A large workflow marketplace
- Complex enterprise policy language
- Fully autonomous financial or destructive actions
- Making the animated interface mandatory

## 16. Roadmap

### Phase 1: Reliable local host

- Desktop/web command center
- Local Docker sandbox adapter
- Agent lifecycle and event model
- Plain-English progress timeline
- Git diffs, tests, and completion receipts
- Start, pause, resume, retry, and cancel

### Phase 2: Cloud continuation and GPU

- Modal adapter
- Remote supervisor
- Automatic checkpoint uploads
- Continue running after laptop disconnects
- Cost estimates and spending limits
- Automatic local-versus-cloud routing

### Phase 3: Universal connection layer

- Remote MCP server
- Python and TypeScript SDKs
- E2B or equivalent desktop/browser adapter
- Daytona or equivalent persistent VM adapter
- AWS and Azure account connections
- Additional agents and models

### Phase 4: Collaboration and evaluation

- Multi-user projects
- Task assignment and handoff
- Isolated agent branches and conflict handling
- Reusable evaluator scenarios
- Comparative agent runs
- Team policies and audit exports

### Phase 5: Ecosystem

- Workflow and environment marketplace
- Provider marketplace and routing optimization
- Enterprise BYOC deployments
- Organization-wide permissions
- Advanced visual themes and character system
- Public and private workflow benchmarks

## 17. Business model

Possible packaging:

- **Free:** Local execution, one user, basic supervision
- **Pro:** Cloud continuation, GPU routing, more concurrent agents, history, and advanced checkpoints
- **Team:** Shared projects, task assignment, reviews, budgets, and collaboration
- **Enterprise:** BYOC, centralized policy, audit exports, SSO, data residency, and support

Cloud compute should be clearly itemized. The platform can charge a subscription plus either pass-through compute or a transparent orchestration margin.

## 18. Success metrics

The most important early metrics are not total sandbox launches. They are whether users confidently delegate and receive successful results.

- Time from signup to first successful agent run
- Percentage of missions completed with evidence
- Percentage of interrupted missions successfully resumed
- Median time users spend supervising each mission
- Rate of false or unsupported completion claims
- Percentage of users returning for another mission
- Number of collaborative projects
- Compute cost per successful mission
- Percentage of automatically routed jobs accepted without override
- User-reported understanding of current agent state

## 19. Immediate product decisions

Before implementation, decide:

1. Which first two agent clients will be supported?
2. Will the initial application be desktop-first, web-first, or a desktop application with a cloud supervisor?
3. Which jobs may migrate automatically, and which require approval?
4. What exact evidence is required before a coding task is marked complete?
5. Which Modal GPU and CPU workflows belong in the first demonstration?
6. What is the simplest checkpoint format that can survive provider changes?
7. Which MCP operations are read-only, writable, destructive, or billable?
8. What spending limit is enabled by default?
9. How will the simple timeline be generated from structured runtime events?
10. Which visual scene best demonstrates real progress without concealing technical truth?

## 20. One-sentence product definition

**A consumer-friendly command center where people and AI systems can launch, supervise, pause, resume, verify, and collaborate with agents running across local computers and cloud compute through one secure interface.**
