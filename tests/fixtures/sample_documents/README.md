# Sample Documents for Testing

This directory contains sample documents used for testing the RAG pipeline.

## Contents

- `sample.txt` - A minimal text document for basic testing
- `chinese_chunk_validation.md` - A long-form Chinese technical document for dashboard-based chunk validation

## Usage

These documents are used by integration and e2e tests to verify:
- Document loading
- Text splitting
- Embedding generation
- Vector storage operations

## Recommended Manual Validation

Use `chinese_chunk_validation.md` when you want to verify chunk boundaries with the default
ingestion configuration (`chunk_size=1000`, `chunk_overlap=200`) through the Dashboard upload flow.

Expected observations after upload:
- The ingestion trace should show multiple chunks instead of a single fragment.
- Chunk previews should mostly break on section or paragraph boundaries.
- The Data Browser should show readable chunk bodies and non-empty metadata summaries when enrichment is enabled.
