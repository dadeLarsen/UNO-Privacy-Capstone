// Client for the localhost API in ../api. The browser never calls modelService directly.

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8001'

export type Category = {
  id: number
  name: string
  display_name: string
  definition: string
  color: string
  tint?: string | null
}

export type CategoryList = {
  taxonomy_id: string
  version: number
  categories: Category[]
}

export type ModelProfile = {
  id: string
  revision: string
  precision: string
  device: string
  strategy: string
  head?: string | null
  fit?: string | null
  taxonomy_id?: string | null
  taxonomy_version?: number | null
  prompt?: string | null
  prompt_version?: number | null
}

export type Sentence = {
  id: number
  text: string
  start: number
  end: number
  label_id: number | null
  label: string | null
  scores?: number[] | null
  error?: string | null
}

export type CategoryCount = { label_id: number; name: string; count: number }

export type Classification = {
  policy_id: string
  model: ModelProfile
  taxonomy_id: string
  taxonomy_version: number
  sentences: Sentence[]
  counts: CategoryCount[]
}

export type Summary = {
  label_id: number
  category: string
  text: string
  source_sentence_ids: number[]
}

export type Rewrite = {
  id: string
  text: string
  source_sentence_ids: number[]
  warnings: string[]
}

export type SummaryResult = {
  kind: 'summary'
  policy_id: string
  model: ModelProfile
  taxonomy_id: string
  taxonomy_version: number
  summaries: Summary[]
}

export type RewriteResult = {
  kind: 'rewrite'
  policy_id: string
  model: ModelProfile
  style: RewriteStyle
  rewrites: Rewrite[]
}

export type Job = {
  id: string
  policy_id: string
  kind: 'summary' | 'rewrite'
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'
  progress: { stage: string; completed: number; total: number; message?: string | null } | null
  created_at: string
  updated_at: string
  error?: { code: string; message: string; retryable: boolean } | null
  // A rewrite job's sections in the order it rewrites them, each as the IDs of its sentences.
  sections?: number[][] | null
  result?: SummaryResult | RewriteResult | null
}

export type Policy = {
  id: string
  source: 'paste' | 'upload' | 'example'
  filename?: string | null
  text_sha256: string
  characters: number
  created_at: string
}

export type AnalysisSnapshot = {
  policy: Policy
  phase: 'awaiting_classification' | 'classified' | 'generating' | 'results_ready'
  classification: Classification | null
  summaries: Summary[]
  rewrites: Rewrite[]
  updated_at: string
}

export type RewriteStyle = 'paraphrase' | 'simplify_for_kids' | 'simplify_for_adults'

// What the page calls each style.
export const styleLabels: Record<RewriteStyle, string> = {
  paraphrase: 'Paraphrase',
  simplify_for_kids: 'Simplify for Kids',
  simplify_for_adults: 'Simplify for Adults',
}

type Detail = string | { code: string; message: string; retryable: boolean } | undefined

// `code` is the API's stable error code, such as 'not_found'; it is absent when the answer had none.
export class ApiError extends Error {
  code: string | undefined

  constructor(message: string, code?: string) {
    super(message)
    this.code = code
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    let detail: Detail
    try {
      detail = (await response.json()).detail as Detail
    } catch {
      detail = undefined
    }
    const message = typeof detail === 'string' ? detail : detail?.message
    const code = typeof detail === 'string' ? undefined : detail?.code
    throw new ApiError(message ?? `Request failed with HTTP ${response.status}`, code)
  }
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

function jsonInit(method: string, body: unknown): RequestInit {
  return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
}

export function getCategories(): Promise<CategoryList> {
  return request<CategoryList>('/api/v1/categories')
}

export function createPolicy(text: string): Promise<Policy> {
  return request<Policy>('/api/v1/policies', jsonInit('POST', { text, source: 'paste' }))
}

export function uploadPolicy(file: File): Promise<Policy> {
  const form = new FormData()
  form.append('file', file)
  return request<Policy>('/api/v1/policies/upload', { method: 'POST', body: form })
}

export function classify(policyId: string, includeScores = false): Promise<Classification> {
  return request<Classification>(
    `/api/v1/policies/${policyId}/classification`,
    jsonInit('POST', { include_scores: includeScores }),
  )
}

export function startSummaries(
  policyId: string,
  body: { category_ids?: number[]; sentence_ids?: number[] } = {},
): Promise<Job> {
  return request<Job>(`/api/v1/policies/${policyId}/summaries`, jsonInit('POST', body))
}

export function startRewrite(
  policyId: string,
  style: RewriteStyle,
  sentenceIds?: number[],
): Promise<Job> {
  return request<Job>(
    `/api/v1/policies/${policyId}/rewrites`,
    jsonInit('POST', { style, ...(sentenceIds ? { sentence_ids: sentenceIds } : {}) }),
  )
}

export function getJob(jobId: string): Promise<Job> {
  return request<Job>(`/api/v1/jobs/${jobId}`)
}

export function cancelJob(jobId: string): Promise<void> {
  return request<void>(`/api/v1/jobs/${jobId}`, { method: 'DELETE' })
}

export function getResults(policyId: string): Promise<AnalysisSnapshot> {
  return request<AnalysisSnapshot>(`/api/v1/policies/${policyId}/results`)
}
