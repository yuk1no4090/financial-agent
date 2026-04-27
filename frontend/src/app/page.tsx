import {
  ArrowRightIcon,
  BarChart3Icon,
  Building2Icon,
  FileSearchIcon,
  LineChartIcon,
  NewspaperIcon,
  ShieldAlertIcon,
  SparklesIcon,
  WorkflowIcon,
} from "lucide-react";
import Link from "next/link";

import { Button } from "@/components/ui/button";

const quickTasks = [
  {
    title: "Earnings Analysis",
    body: "Parse filings, earnings calls, margins, cash flow and management guidance.",
    icon: FileSearchIcon,
  },
  {
    title: "Peer Comparison",
    body: "Compare revenue growth, valuation multiples, balance-sheet quality and moat.",
    icon: Building2Icon,
  },
  {
    title: "Risk Review",
    body: "Surface accounting, liquidity, regulatory, competitive and macro risks.",
    icon: ShieldAlertIcon,
  },
  {
    title: "Market Brief",
    body: "Turn news, events and price moves into a concise analyst-ready brief.",
    icon: NewspaperIcon,
  },
];

const workflow = [
  "Collect market data, filings and news",
  "Reason with a finance-tuned model",
  "Coordinate specialist sub-agents",
  "Generate an investment memo with sources",
];

const prompts = [
  "Analyze Apple's latest earnings and highlight revenue drivers, margin pressure, catalysts and risks.",
  "Compare NVIDIA and AMD across growth, valuation, gross margin trend and competitive positioning.",
  "Build an investment memo for Tesla using recent filings, news, valuation assumptions and risk factors.",
];

export default function LandingPage() {
  return (
    <main className="min-h-screen overflow-hidden bg-[#f3f1ea] text-[#171a16]">
      <section className="relative min-h-screen border-b border-[#d6d0c4]">
        <div className="absolute inset-0 bg-[linear-gradient(120deg,rgba(21,70,52,0.12),transparent_36%),linear-gradient(0deg,rgba(255,255,255,0.72),rgba(255,255,255,0.2)),radial-gradient(circle_at_78%_18%,rgba(225,176,74,0.28),transparent_28%)]" />
        <div className="absolute inset-x-0 bottom-0 h-40 bg-[linear-gradient(0deg,rgba(23,26,22,0.1),transparent)]" />
        <div className="relative mx-auto flex min-h-screen max-w-7xl flex-col px-5 py-5 sm:px-8 lg:px-10">
          <header className="flex items-center justify-between">
            <Link
              href="/"
              className="flex items-center gap-3 text-sm font-semibold tracking-wide"
            >
              <span className="flex size-9 items-center justify-center rounded-md bg-[#173f34] text-white">
                FA
              </span>
              <span>FinAgent Workbench</span>
            </Link>
            <nav className="hidden items-center gap-6 text-sm text-[#5f6359] md:flex">
              <a href="#workflow" className="hover:text-[#173f34]">
                Workflow
              </a>
              <a href="#tasks" className="hover:text-[#173f34]">
                Analyst Tasks
              </a>
              <a href="#prompts" className="hover:text-[#173f34]">
                Prompts
              </a>
            </nav>
            <Button
              asChild
              className="bg-[#173f34] text-white hover:bg-[#215946]"
            >
              <Link href="/workspace">
                Open Desk
                <ArrowRightIcon className="size-4" />
              </Link>
            </Button>
          </header>

          <div className="grid flex-1 items-center gap-10 py-16 lg:grid-cols-[1.02fr_0.98fr] lg:py-8">
            <div className="max-w-3xl">
              <div className="mb-6 inline-flex items-center gap-2 rounded-md border border-[#c8bea9] bg-white/55 px-3 py-2 text-xs font-medium tracking-[0.18em] text-[#596354] uppercase">
                <SparklesIcon className="size-3.5 text-[#b7791f]" />
                Agentic Financial Analysis
              </div>
              <h1 className="max-w-4xl text-5xl leading-[1.02] font-semibold tracking-normal text-[#11140f] sm:text-6xl lg:text-7xl">
                A research desk for financial agents.
              </h1>
              <p className="mt-6 max-w-2xl text-lg leading-8 text-[#54584f]">
                Coordinate domain-tuned models, market data tools and specialist
                sub-agents to produce investment research reports, earnings
                reviews and risk memos from a single workspace.
              </p>
              <div className="mt-8 flex flex-col gap-3 sm:flex-row">
                <Button
                  asChild
                  size="lg"
                  className="bg-[#173f34] text-white hover:bg-[#215946]"
                >
                  <Link href="/workspace/chats/new">
                    Start Financial Analysis
                    <ArrowRightIcon className="size-4" />
                  </Link>
                </Button>
                <Button
                  asChild
                  size="lg"
                  variant="outline"
                  className="border-[#b8ad9b] bg-white/45 hover:bg-white/80"
                >
                  <Link href="/workspace/agents">View Agent Roles</Link>
                </Button>
              </div>
            </div>

            <div className="rounded-lg border border-[#cfc5b4] bg-[#101511] p-3 shadow-2xl shadow-[#173f34]/20">
              <div className="rounded-md border border-white/10 bg-[#161d18] p-4">
                <div className="flex items-center justify-between border-b border-white/10 pb-4">
                  <div>
                    <p className="text-xs tracking-[0.2em] text-[#9da895] uppercase">
                      Live Analyst Run
                    </p>
                    <h2 className="mt-1 text-xl font-semibold text-white">
                      NVDA Earnings Review
                    </h2>
                  </div>
                  <div className="rounded-md bg-[#d9a441] px-3 py-1 text-xs font-semibold text-[#15100a]">
                    Draft
                  </div>
                </div>

                <div className="grid gap-3 py-4 sm:grid-cols-3">
                  <Metric label="Revenue Growth" value="+94%" tone="green" />
                  <Metric label="Gross Margin" value="75.0%" tone="gold" />
                  <Metric label="Forward P/E" value="32.8x" tone="muted" />
                </div>

                <div className="grid gap-3 lg:grid-cols-[1fr_0.75fr]">
                  <div className="rounded-md border border-white/10 bg-white/[0.04] p-4">
                    <div className="mb-4 flex items-center gap-2 text-sm font-medium text-white">
                      <LineChartIcon className="size-4 text-[#7cc6a4]" />
                      Thesis Timeline
                    </div>
                    <div className="space-y-3">
                      {[
                        [
                          "Data Agent",
                          "Collected filings, transcript and news",
                        ],
                        [
                          "Accounting Agent",
                          "Checked margin, cash conversion and capex",
                        ],
                        [
                          "Risk Agent",
                          "Flagged export controls and supply concentration",
                        ],
                        ["Report Agent", "Composing memo with citations"],
                      ].map(([agent, detail]) => (
                        <div
                          className="grid grid-cols-[7rem_1fr] gap-3 text-sm"
                          key={agent}
                        >
                          <span className="text-[#9da895]">{agent}</span>
                          <span className="text-[#f2eee3]">{detail}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="rounded-md border border-white/10 bg-[#20281f] p-4">
                    <div className="mb-4 flex items-center gap-2 text-sm font-medium text-white">
                      <BarChart3Icon className="size-4 text-[#d9a441]" />
                      Output
                    </div>
                    <div className="space-y-2 text-sm text-[#d7d2c4]">
                      <p>Executive summary</p>
                      <p>Financial performance</p>
                      <p>Valuation bridge</p>
                      <p>Risks and catalysts</p>
                      <p>Source-backed memo</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="workflow" className="border-b border-[#d6d0c4] bg-[#faf8f1]">
        <div className="mx-auto grid max-w-7xl gap-8 px-5 py-16 sm:px-8 lg:grid-cols-[0.75fr_1.25fr] lg:px-10">
          <div>
            <p className="text-sm font-semibold tracking-[0.18em] text-[#8a6b25] uppercase">
              Workflow
            </p>
            <h2 className="mt-3 text-3xl font-semibold">
              Built for analyst loops, not generic chat.
            </h2>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {workflow.map((item, index) => (
              <div
                key={item}
                className="rounded-md border border-[#d9d0bd] bg-white p-5"
              >
                <div className="mb-5 flex size-8 items-center justify-center rounded-md bg-[#173f34] text-sm font-semibold text-white">
                  {index + 1}
                </div>
                <p className="text-base font-medium">{item}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="tasks" className="bg-[#f3f1ea]">
        <div className="mx-auto max-w-7xl px-5 py-16 sm:px-8 lg:px-10">
          <div className="mb-8 flex items-end justify-between gap-6">
            <div>
              <p className="text-sm font-semibold tracking-[0.18em] text-[#8a6b25] uppercase">
                Analyst Tasks
              </p>
              <h2 className="mt-3 text-3xl font-semibold">
                Presets for financial research.
              </h2>
            </div>
            <WorkflowIcon className="hidden size-10 text-[#173f34] md:block" />
          </div>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {quickTasks.map((task) => (
              <div
                className="rounded-md border border-[#d6d0c4] bg-white/80 p-5"
                key={task.title}
              >
                <task.icon className="mb-6 size-6 text-[#173f34]" />
                <h3 className="text-lg font-semibold">{task.title}</h3>
                <p className="mt-3 text-sm leading-6 text-[#5f6359]">
                  {task.body}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="prompts" className="bg-[#151a16] text-white">
        <div className="mx-auto max-w-7xl px-5 py-16 sm:px-8 lg:px-10">
          <div className="grid gap-8 lg:grid-cols-[0.75fr_1.25fr]">
            <div>
              <p className="text-sm font-semibold tracking-[0.18em] text-[#d9a441] uppercase">
                Starter Prompts
              </p>
              <h2 className="mt-3 text-3xl font-semibold">
                Ready to send into the workspace.
              </h2>
            </div>
            <div className="space-y-3">
              {prompts.map((prompt) => (
                <Link
                  className="block rounded-md border border-white/10 bg-white/[0.04] p-4 text-sm leading-6 text-[#e9e2d2] transition hover:border-[#d9a441]/60 hover:bg-white/[0.08]"
                  href={`/workspace/chats/new?initialPrompt=${encodeURIComponent(prompt)}`}
                  key={prompt}
                >
                  {prompt}
                </Link>
              ))}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}

function Metric({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "green" | "gold" | "muted";
}) {
  const color =
    tone === "green"
      ? "text-[#7cc6a4]"
      : tone === "gold"
        ? "text-[#d9a441]"
        : "text-[#d7d2c4]";
  return (
    <div className="rounded-md border border-white/10 bg-white/[0.04] p-3">
      <p className="text-[0.68rem] tracking-[0.16em] text-[#9da895] uppercase">
        {label}
      </p>
      <p className={`mt-2 text-2xl font-semibold ${color}`}>{value}</p>
    </div>
  );
}
