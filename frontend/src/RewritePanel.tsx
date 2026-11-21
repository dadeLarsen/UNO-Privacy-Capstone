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
  activeJob: Job | null
  rewrites: Rewrite[]
  onRewrite: (style: RewriteStyle, label: string) => void
  onSummarize: () => void
  onCancel: () => void
}

function RewritePanel({ busy, activeJob, rewrites, onRewrite, onSummarize, onCancel }: Props) {
  const [style, setStyle] = useState<RewriteStyle>('simplify_for_adults')

  return (
    <section className="rewrite-panel">
      <div className="panel-heading">
        <div><span className="step">03</span><h2>Make it clearer</h2></div>
      </div>
      <p className="rewrite-copy">
        The rewriter is wired in and runs locally. Choose a reading style and select sentences
        above, or rewrite the whole policy. Until style-specific prompts are evaluated, every
        style uses the benchmark's <strong>summary-v3</strong> prompt; warnings flag details to
        check.
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
          <button className="secondary-button" onClick={onCancel} type="button">
            Cancel {activeJob.progress?.stage ?? activeJob.status}
          </button>
        )}
      </div>
      <div className="rewrite-result">
        <span className="result-label">
          {rewrites.length ? 'Latest rewrite' : 'No rewrite generated yet'}
        </span>
        {rewrites.length ? (
          rewrites.map((item) => (
            <div key={item.id}>
              <p>{item.text}</p>
              <span className="source-link">
                Sources: {item.source_sentence_ids.map((id) => id + 1).join(', ')}
              </span>
              {item.warnings.length ? (
                <ul className="rewrite-warnings">
                  {item.warnings.map((warning) => <li key={warning}>{warning}</li>)}
                </ul>
              ) : null}
            </div>
          ))
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
