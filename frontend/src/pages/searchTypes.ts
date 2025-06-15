export interface SearchDocument {
  id: string;
  title: string;
  preview: string;
  project_id: string;
  updated_at: string;
  type: string;
}

export interface SearchChat {
  id: string;
  thread_id: string;
  preview: string;
  created_at: string;
  type: string;
}

export interface SearchSemanticMatch {
  id: string;
  content: string;
  score: number;
  metadata: Record<string, unknown>; // Or a more specific type if known
  type: string;
}

export interface SearchResult {
  documents: SearchDocument[];
  chats: SearchChat[];
  semantic_matches: SearchSemanticMatch[];
}
