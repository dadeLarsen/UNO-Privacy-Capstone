import { useState, type CSSProperties } from 'react'
import type { Category, Sentence } from './api'

type Props = {
  sentences: Sentence[]
  // False until the API has labeled the sentences. They cannot be selected before that.
  analyzed: boolean
  categories: Map<number, Category>
  activeCategories: Set<number>
  selected: Set<number>
  onToggle: (id: number) => void
}

function tintOf(category: Category): string {
  return category.tint ?? `${category.color}1a`
}

function SentenceMap({ sentences, analyzed, categories, activeCategories, selected, onToggle }: Props) {
  const [highlighted, setHighlighted] = useState(true)

  return (
    <>
      <div className="panel-heading result-heading">
        <div><span className="step">02</span><h2>Sentence map</h2></div>
        <label className="toggle-label">
          <input
            checked={highlighted && analyzed}
            disabled={!analyzed}
            onChange={(event) => setHighlighted(event.target.checked)}
            type="checkbox"
          />
          <span className="toggle" /> Highlight categories
        </label>
      </div>
      {analyzed && (
        <p className="selection-hint">
          {selected.size
            ? `${selected.size} sentence${selected.size === 1 ? '' : 's'} selected for rewriting. Click again to deselect.`
            : 'Click sentences to select them for rewriting. With none selected, the whole policy is used.'}
        </p>
      )}
      <div className="sentence-list">
        {sentences.map((sentence) => {
          const category = sentence.label_id === null ? undefined : categories.get(sentence.label_id)
          const isVisible = highlighted && category !== undefined && activeCategories.has(sentence.label_id!)
          const isSelected = selected.has(sentence.id)
          return (
            <div
              aria-pressed={analyzed ? isSelected : undefined}
              className={`sentence ${isVisible ? 'is-highlighted' : ''} ${isSelected ? 'is-selected' : ''}`}
              key={sentence.id}
              onClick={() => onToggle(sentence.id)}
              onKeyDown={(event) => {
                if (analyzed && (event.key === 'Enter' || event.key === ' ')) {
                  event.preventDefault()
                  onToggle(sentence.id)
                }
              }}
              role={analyzed ? 'button' : undefined}
              style={
                category
                  ? { '--sentence-tint': tintOf(category), '--sentence-color': category.color } as CSSProperties
                  : undefined
              }
              tabIndex={analyzed ? 0 : undefined}
            >
              <span className="sentence-number">{String(sentence.id + 1).padStart(2, '0')}</span>
              <p>{sentence.text}</p>
              {category && <span className="sentence-tag">{category.display_name}</span>}
              {sentence.error && <span className="sentence-error">{sentence.error}</span>}
            </div>
          )
        })}
        {!sentences.length && <div className="empty-state">Paste policy text above to see sentence-level results.</div>}
      </div>
    </>
  )
}

export default SentenceMap
