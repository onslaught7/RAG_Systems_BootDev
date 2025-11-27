# Technical Documentation - CLI Module

## Overview

The CLI (Command-Line Interface) module provides a user-friendly command-line interface for interacting with the RAG search engine. It enables users to build inverted indexes from movie data and perform keyword-based searches through a simple, intuitive command-line interface.

## Architecture

The CLI module consists of a single entry point (`keyword_search_cli.py`) that provides two main commands:

1. **`build`** - Constructs and persists the inverted index from movie data
2. **`search`** - Performs keyword searches against the built index

The CLI acts as a thin wrapper around the core search engine modules (`src.inverted_index` and `src.text_processing`), providing a convenient interface for end users.

## Module Details

### `keyword_search_cli.py`

This module implements the command-line interface using Python's `argparse` library for argument parsing and subcommand handling.

#### Path Configuration

The module dynamically adds the parent directory to `sys.path` to enable imports from the `src/` package:

```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

This allows the CLI to import modules from the `src/` directory regardless of the execution context.

#### Key Functions

##### `search_movies(query: str) -> List[str]`

Performs a keyword search operation and returns matching movie titles.

**Parameters:**
- `query` (str): The search query string

**Returns:**
- `List[str]`: List of movie titles matching the query (up to 5 results)

**Process Flow:**
1. Initializes an `InvertedIndex` instance
2. Attempts to load the index from disk (cached index)
3. Normalizes the query text using `_normalize_text()`
4. For each token in the normalized query (limited to first 5 tokens):
   - Retrieves matching document IDs from the index
   - Accumulates results in a set to avoid duplicates
5. Maps document IDs to movie titles using the docmap
6. Returns top 5 results sorted by document ID

**Error Handling:**
- If the index is not found, prints an error message and returns an empty list
- Handles cases where normalized query is empty

**Performance Considerations:**
- Limits query processing to first 5 tokens for efficiency
- Uses set operations for O(1) document ID lookups
- Returns maximum of 5 results to keep output manageable

##### `main() -> None`

The main entry point for the CLI application. Sets up argument parsing and routes commands to appropriate handlers.

**Command Structure:**
- Uses `argparse.ArgumentParser` with subparsers for command handling
- Implements a subcommand pattern for extensibility

**Commands:**

1. **`search <query>`**
   - Searches for movies matching the query string
   - Displays results in a numbered list format
   - Shows "No results found" if search returns empty

2. **`build`**
   - Builds the inverted index from `./data/movies.json`
   - Automatically saves the index to `./cache/` directory
   - No additional arguments required

3. **Default (no command or invalid command)**
   - Displays help message with available commands

**Output Format:**
- Search command: Prints query, then numbered list of results
- Build command: Silent execution (errors printed if they occur)
- Help: Standard argparse help output

## Command-Line Interface

### Usage

```bash
python cli/keyword_search_cli.py <command> [arguments]
```

### Commands

#### Build Index

```bash
python cli/keyword_search_cli.py build
```

- Builds the inverted index from `./data/movies.json`
- Saves index to `./cache/index.pkl` and `./cache/docmap.pkl`
- Creates cache directory if it doesn't exist
- Must be run before performing searches

#### Search Movies

```bash
python cli/keyword_search_cli.py search "<query>"
```

- Performs keyword search on the built index
- Query is normalized (lowercased, stemmed, stop words removed)
- Returns up to 5 matching movie titles
- Results are sorted by document ID

**Example:**
```bash
python cli/keyword_search_cli.py search "action adventure"
```

**Output:**
```
Searching for: action adventure
1. The Matrix
2. Inception
3. Interstellar
...
```

### Help

```bash
python cli/keyword_search_cli.py --help
python cli/keyword_search_cli.py search --help
python cli/keyword_search_cli.py build --help
```

## Dependencies

### External Dependencies
- `argparse` - Standard library for command-line argument parsing
- `json` - Standard library for JSON handling (used by InvertedIndex)
- `pathlib` - Standard library for path manipulation

### Internal Dependencies
- `src.text_processing._normalize_text` - Text normalization function
- `src.inverted_index.InvertedIndex` - Core inverted index implementation

## Integration with Core Modules

The CLI module integrates with the core search engine as follows:

```
CLI (keyword_search_cli.py)
    ↓
    ├─→ InvertedIndex (src/inverted_index.py)
    │   ├─→ build() - Index construction
    │   ├─→ load() - Index loading
    │   └─→ get_documents() - Document retrieval
    │
    └─→ _normalize_text() (src/text_processing.py)
        └─→ Text preprocessing and normalization
```

### Data Flow

**Build Command:**
```
CLI → InvertedIndex.build() → Read movies.json → Index documents → Save to cache
```

**Search Command:**
```
CLI → Load index → Normalize query → Token lookup → Aggregate results → Display titles
```

## Error Handling

The CLI handles various error scenarios:

1. **Missing Index Files**
   - When `search` command is executed without a built index
   - Error message: "Error: Inverted index not found. Please build the index first using the 'build' command."
   - Returns empty results list

2. **Missing Data Files**
   - Handled by `InvertedIndex.build()` method
   - Error message: "Error: ../data/movies.json not found."

3. **Empty Queries**
   - Normalized query returns empty list
   - Returns empty results without error

4. **Invalid Commands**
   - Displays help message when no command or invalid command provided

## Design Decisions

1. **Subcommand Pattern**: Uses argparse subparsers for clean command separation and extensibility
2. **Result Limiting**: Limits to 5 tokens and 5 results for performance and usability
3. **Path Management**: Dynamically adjusts Python path to support flexible execution contexts
4. **User-Friendly Output**: Provides clear, numbered results with descriptive error messages
5. **Separation of Concerns**: CLI focuses on user interaction, delegates core logic to src modules

## Performance Considerations

- **Lazy Index Loading**: Index is only loaded when search command is executed
- **Token Limiting**: Processes only first 5 query tokens to maintain performance
- **Result Capping**: Returns maximum 5 results to keep response times reasonable
- **Set Operations**: Uses set union for efficient document ID aggregation

## Future Enhancements

Potential improvements could include:

- **BM25 Ranking**: Implement BM25 algorithm (currently mentioned in help text but not implemented)
- **Output Formatting**: Add options for JSON, CSV, or other output formats
- **Verbose Mode**: Add `--verbose` flag for detailed search information
- **Query Options**: Support for boolean operators (AND, OR, NOT)
- **Pagination**: Support for retrieving more than 5 results
- **Interactive Mode**: REPL-style interface for multiple searches
- **Configuration**: Support for custom data paths, cache locations, etc.
- **Progress Indicators**: Show progress during index building for large datasets

## File Structure

```
cli/
├── keyword_search_cli.py    # Main CLI implementation
└── TECHNICAL_DOCS.md        # This file
```

## Testing Considerations

When testing the CLI:

1. **Build Command**: Verify index creation and cache file generation
2. **Search Command**: Test with various query types (single word, multi-word, special characters)
3. **Error Cases**: Test missing index, missing data files, empty queries
4. **Edge Cases**: Test with very long queries, queries with only stop words
5. **Integration**: Verify proper interaction with src modules