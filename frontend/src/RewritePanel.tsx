import { useState } from 'react'
import type { Job, Rewrite, RewriteStyle } from './api'

const styleLabels: Record<RewriteStyle, string> = {
  paraphrase: 'Paraphrase',
  simplify_for_kids: 'Simplify for Kids',
  simplify_for_adults: 'Simplify for Adults',
}

const styleIcons: Record<RewriteStyle, string> = {
  paraphrase: '↗',
  simplify_for_kids: '☼',
  simplify_for_adults: '◌',
}

type Props = {
  busy: boolean
  // What Generate rewrites: the selection or the whole policy.
  scope: string
  // The job that runs now, of either kind.
  activeJob: Job | null
  // The latest rewrite job as last polled, also after it ended.
  rewriteJob: Job | null
  // The rewrite in the policy's results.
  rewrites: Rewrite[]
  onRewrite: (style: RewriteStyle, label: string) => void
  onSummarize: () => void
  onCancel: () => void
}

// How a rewrite job that is not whole heads its sections; null for a whole rewrite or one with no section yet.
function partLabel(job: Job | null): string | null {
  if (!job?.progress || job.status === 'succeeded' || job.progress.completed === 0) return null
  const ending = job.status === 'cancelled' ? 'cancelled' : job.status === 'failed' ? 'failed' : 'so far'
  return `Rewrite ${ending} · ${job.progress.completed} of ${job.progress.total} sections`
}

function RewritePanel({ busy, scope, activeJob, rewriteJob, rewrites, onRewrite, onSummarize, onCancel }: Props) {
  const [style, setStyle] = useState<RewriteStyle>('simplify_for_adults')
  // A rewrite job holds its finished sections while it runs and after it ends. The results hold the
  // same sections once it ends, and an earlier rewrite when the job ended before its first section.
  const sections = rewriteJob?.result?.kind === 'rewrite' ? rewriteJob.result.rewrites : rewrites
  // A warning on every section, such as the note on the style prompt, shows once above them.
  const shared = sections.length
    ? sections[0].warnings.filter((warning) => sections.every((item) => item.warnings.includes(warning)))
    : []
  const progress = activeJob?.progress

  return (
    <section className="rewrite-panel">
      <div className="panel-heading">
        <div><span className="step">03</span><h2>Make it clearer</h2></div>
      </div>
      <p className="rewrite-copy">
        The rewriter is wired in and runs locally. Choose a reading style and select sentences
        above, or rewrite the whole policy, which goes one section at a time. Until style-specific
        prompts are evaluated, every style uses the benchmark's <strong>summary-v3</strong> prompt;
        warnings flag details to check.
      </p>
      <div className="rewrite-options">
        {(Object.keys(styleLabels) as RewriteStyle[]).map((option) => (
          <button
            className={style === option ? 'selected' : ''}
            onClick={() => setStyle(option)}
            key={option}
            type="button"
          >
            <span>{styleIcons[option]}</span>{styleLabels[option]}
          </button>
        ))}
      </div>
      <div className="rewrite-actions">
        <p className="rewrite-scope">Scope: {scope}</p>
        <button
          className="primary-button"
          onClick={() => onRewrite(style, styleLabels[style])}
          disabled={busy}
          type="button"
        >
          <span>Generate {styleLabels[style]}</span><b>→</b>
        </button>
        <button className="secondary-button" onClick={onSummarize} disabled={busy} type="button">
          Summarize categories
        </button>
        {activeJob && (
          <div className="rewrite-progress">
            <span role="status">{progress?.message ?? 'Working…'}</span>
            {progress && progress.total > 1 && <progress max={progress.total} value={progress.completed} />}
            <button className="secondary-button" onClick={onCancel} type="button">
              Cancel {activeJob.kind}
            </button>
          </div>
        )}
      </div>
      <div className="rewrite-result">
        <span className="result-label">
          {sections.length ? partLabel(rewriteJob) ?? 'Latest rewrite' : 'No rewrite generated yet'}
        </span>
        {shared.length ? (
          <ul className="rewrite-warnings">
            {shared.map((warning) => <li key={warning}>{warning}</li>)}
          </ul>
        ) : null}
        {sections.length ? (
          sections.map((item) => {
            const own = item.warnings.filter((warning) => !shared.includes(warning))
            return (
              <div key={item.id}>
                <p>{item.text}</p>
                <span className="source-link">
                  Sources: {item.source_sentence_ids.map((id) => id + 1).join(', ')}
                </span>
                {own.length ? (
                  <ul className="rewrite-warnings">
                    {own.map((warning) => <li key={warning}>{warning}</li>)}
                  </ul>
                ) : null}
              </div>
            )
          })
        ) : (
          <p>
            Analyze a policy, choose a style, then generate a source-linked rewrite. The
            rewriter is available locally; if its one-time model download is missing, the job
            reports the exact setup command.
          </p>
        )}
      </div>
    </section>
  )
}

export default RewritePanel
