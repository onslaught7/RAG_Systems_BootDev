# Technical Documentation

## Overview

This RAG (Retrieval-Augmented Generation) search engine implements a keyword-based search system for movie data using an inverted index data structure. The system provides efficient text search capabilities through tokenization, normalization, and indexing.

## Architecture

The system consists of three main components:

1. **Text Processing Module** (`text_processing.py`) - Handles text normalization and preprocessing
2. **Inverted Index Module** (`inverted_index.py`) - Manages the inverted index data structure and document mapping
3. **CLI Interface** (`cli/keyword_search_cli.py`) - Provides command-line interface for building and searching

## Module Details

### `text_processing.py`

This module handles all text preprocessing operations required for keyword search.

#### Key Functions

- **`_load_stop_words() -> set`**
  - Loads stop words from `./data/stopwords.txt`
  - Uses `@lru_cache` decorator for performance optimization
  - Returns an empty set if the file is not found

- **`_normalize_text(text: str) -> List[str]`**
  - Performs comprehensive text normalization:
    1. **Case insensitivity**: Converts text to lowercase
    2. **Punctuation removal**: Removes all punctuation characters
    3. **Tokenization**: Splits text into individual words
    4. **Stop words removal**: Filters out common stop words
    5. **Stemming**: Applies Porter Stemmer algorithm to reduce words to their root forms
  - Returns a list of normalized tokens
  - Returns empty list if input is not a string

#### Dependencies
- `nltk.stem.PorterStemmer` - For word stemming
- `functools.lru_cache` - For caching stop words
- `string` - For punctuation handling

### `inverted_index.py`

This module implements the core inverted index data structure for efficient document retrieval.

#### Class: `InvertedIndex`

##### Data Structures

- **`index: Dict[str, Set[int]]`**
  - Maps normalized tokens to sets of document IDs
  - Key: normalized token (string)
  - Value: set of document IDs containing that token

- **`docmap: Dict[int, Dict]`**
  - Maps document IDs to their full metadata
  - Key: document ID (integer)
  - Value: complete document dictionary (e.g., movie data)

##### Methods

- **`__init__(self)`**
  - Initializes empty index and docmap dictionaries

- **`_add_document(self, doc_id: int, text: str) -> None`**
  - Private method to add a document to the index
  - Normalizes the input text
  - Adds document ID to the index for each token
  - Creates new sets for tokens not yet in the index

- **`get_documents(self, term: str) -> List[int]`**
  - Retrieves document IDs for a given search term
  - Normalizes the term before lookup
  - Returns sorted list of document IDs (ascending order)
  - Returns empty list if term cannot be normalized

- **`build(self) -> None`**
  - Builds the inverted index from `./data/movies.json`
  - Iterates through all movies in the dataset
  - Combines title and description for indexing
  - Stores full movie metadata in docmap
  - Automatically saves the index after building
  - Handles file not found and general exceptions

- **`save(self) -> None`**
  - Persists the index and docmap to disk
  - Saves to `./cache/index.pkl` and `./cache/docmap.pkl`
  - Creates cache directory if it doesn't exist
  - Uses pickle serialization for efficient storage

- **`load(self) -> None`**
  - Loads the index and docmap from disk
  - Reads from `./cache/index.pkl` and `./cache/docmap.pkl`
  - Handles file not found errors gracefully
  - Restores the index state for search operations

#### Dependencies
- `src.text_processing._normalize_text` - For text normalization
- `json` - For reading movie data
- `pickle` - For serialization
- `os` - For directory operations

## Search Algorithm

The search process follows these steps:

1. **Query Normalization**: The search query is normalized using the same process as document indexing
2. **Token Lookup**: For each token in the normalized query (up to 5 tokens), retrieve matching document IDs
3. **Result Aggregation**: Combine document IDs from all query tokens using set union
4. **Result Retrieval**: Fetch movie titles from docmap for matching document IDs
5. **Result Limiting**: Return top 5 results sorted by document ID

## Data Flow

```
User Query
    ↓
CLI Interface (keyword_search_cli.py)
    ↓
Text Normalization (text_processing.py)
    ↓
Inverted Index Lookup (inverted_index.py)
    ↓
Document Mapping Retrieval
    ↓
Results Display
```

## File Structure

```
src/
├── __init__.py              # Package initialization
├── text_processing.py       # Text normalization and preprocessing
├── inverted_index.py        # Inverted index implementation
└── TECHNICAL_DOCS.md        # This file

cli/
└── keyword_search_cli.py    # Command-line interface

data/
├── movies.json              # Movie dataset
└── stopwords.txt            # Stop words list

cache/
├── index.pkl                # Serialized inverted index
└── docmap.pkl               # Serialized document mapping
```

## Key Design Decisions

1. **Stemming**: Uses Porter Stemmer to improve search recall by matching word variations
2. **Stop Words Removal**: Filters common words to reduce index size and improve relevance
3. **Set-based Indexing**: Uses sets for O(1) lookup and automatic deduplication
4. **Pickle Serialization**: Efficient binary format for fast index loading/saving
5. **Token Limit**: Limits query processing to first 5 tokens for performance
6. **Result Limit**: Returns top 5 results to keep output manageable

## Performance Considerations

- **Caching**: Stop words are cached using `@lru_cache` to avoid repeated file I/O
- **Set Operations**: Uses set union for efficient document ID aggregation
- **Sorted Results**: Returns sorted document IDs for consistent ordering
- **Lazy Loading**: Index is loaded only when needed for search operations

## Error Handling

The system handles various error scenarios:
- Missing data files (`movies.json`, `stopwords.txt`)
- Missing cache files (prompts user to build index)
- General exceptions during build/save/load operations
- Invalid input types (non-string text)

## Future Enhancements

Potential improvements could include:
- BM25 ranking algorithm (mentioned in CLI but not implemented)
- Multi-word phrase matching
- Boolean query operators (AND, OR, NOT)
- Relevance scoring and ranking
- Support for additional document types beyond movies