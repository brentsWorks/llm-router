# LLM Router - Technical Design Document

## Overview

A hybrid LLM router that intelligently selects the optimal language model for a given task based on semantic analysis of user prompts, with LLM-assisted fallback for edge cases. The system combines cost efficiency, latency optimization, and quality matching.

**Development Approach**: Test-Driven Development (TDD) with Red-Green-Refactor cycles

## Architecture

### High-Level Flow
```
User Prompt → Semantic Classification → Model Selection → Fallback (if needed) → Route to Provider
```

### Core Components

#### 1. Semantic Classifier (Primary Route)
- **Purpose**: Fast, accurate classification based on prompt embeddings
- **Technology**: RAG with vector similarity search
- **Components**:
  - Embedding Generator (using lightweight embedding model)
  - Vector Store (ChromaDB/Pinecone for similarity search)
  - Example Dataset (curated prompt examples with labels)
  - Confidence Scorer

#### 2. LLM-Assisted Classifier (Fallback Route)
- **Purpose**: Handle edge cases where semantic classification confidence is low
- **Technology**: Fast, cheap LLM for classification
- **Components**:
  - Classification Prompt Template
  - Confidence Threshold Manager
  - Classification Cache

#### 3. Provider Registry
- **Purpose**: Central repository of available models and their capabilities
- **Schema**:
  ```json
  {
    "provider": "string",
    "model": "string", 
    "capabilities": ["code", "creative", "reasoning", "tool-use"],
    "pricing": {
      "input_tokens_per_1k": "number",
      "output_tokens_per_1k": "number"
    },
    "limits": {
      "context_length": "number",
      "rate_limit": "number",
      "safety_level": "string"
    },
    "performance": {
      "avg_latency_ms": "number",
      "quality_scores": {
        "code": "number",
        "creative": "number", 
        "reasoning": "number",
        "summarization": "number"
      }
    }
  }
  ```

#### 4. Scoring Engine
- **Purpose**: Calculate optimal model based on weighted preferences
- **Scoring Function**:
  ```
  Score = w₁×Quality + w₂×(1/Cost) + w₃×(1/Latency) + w₄×ContextMatch
  ```
- **Components**:
  - Weight Configuration Manager
  - Constraint Validator (hard constraints)
  - Preference Optimizer (soft preferences)

#### 5. Routing Policies
- **Hard Constraints**: Must be satisfied
  - Context length limits
  - Safety requirements
  - Rate limiting
  - Provider availability
- **Soft Preferences**: Optimization targets
  - Cost sensitivity
  - Latency tolerance
  - Quality requirements
  - Provider preferences

## Test-Driven Development Strategy

### Testing Philosophy
1. **Red-Green-Refactor**: Write failing tests first, implement minimal code to pass, then refactor
2. **Test Pyramid**: Unit tests (70%) → Integration tests (20%) → End-to-end tests (10%)
3. **Behavior-Driven**: Tests describe business value and expected behaviors
4. **Fast Feedback**: Tests must be fast and reliable for continuous development

### Test Categories

#### Unit Tests (70%)
- **Purpose**: Test individual components in isolation
- **Scope**: Pure functions, data models, business logic
- **Examples**:
  - Scoring function calculations
  - Classification logic
  - Provider registry operations
  - Constraint validation
- **Tools**: pytest, unittest.mock, hypothesis (property-based testing)

#### Integration Tests (20%)
- **Purpose**: Test component interactions
- **Scope**: Database operations, external API calls, service integrations
- **Examples**:
  - Vector store operations
  - LLM API calls (with mocking)
  - Provider registry with real data
  - End-to-end classification pipeline
- **Tools**: pytest, pytest-asyncio, testcontainers

#### End-to-End Tests (10%)
- **Purpose**: Test complete user workflows
- **Scope**: Full routing decisions from prompt to model selection
- **Examples**:
  - Complete routing pipeline
  - API endpoint testing
  - Performance benchmarks
  - Error handling scenarios
- **Tools**: pytest, httpx, locust (load testing)

### Test Structure

#### Test Organization
```
tests/
├── unit/
│   ├── test_models.py           # Data model tests
│   ├── test_scoring.py          # Scoring engine tests
│   ├── test_classification.py   # Classification logic tests
│   └── test_registry.py         # Provider registry tests
├── integration/
│   ├── test_embeddings.py       # Embedding pipeline tests
│   ├── test_vector_store.py     # Vector store integration
│   ├── test_llm_fallback.py     # LLM classification tests
│   └── test_routing.py          # Full routing tests
├── e2e/
│   ├── test_api.py              # API endpoint tests
│   ├── test_workflows.py        # Complete user workflows
│   └── test_performance.py      # Performance and load tests
├── fixtures/
│   ├── sample_prompts.py        # Test data
│   ├── mock_models.py           # Mock model definitions
│   └── test_embeddings.py       # Pre-computed test embeddings
└── conftest.py                  # Shared test configuration
```

#### Test Data Strategy
- **Fixtures**: Reusable test data for consistent testing
- **Factories**: Generate test data with varied parameters
- **Mock Services**: Simulate external dependencies (LLM APIs, vector stores)
- **Golden Files**: Reference outputs for regression testing

## Data Models (with Test Requirements)

### Prompt Classification
```python
@dataclass
class PromptClassification:
    category: str  # "code", "creative", "qa", "summarization", "tool-use"
    subcategory: Optional[str]
    confidence: float
    embedding: List[float]
    reasoning: Optional[str]  # for LLM-assisted classifications
    
    def __post_init__(self):
        # Validation for TDD
        assert 0.0 <= self.confidence <= 1.0
        assert self.category in VALID_CATEGORIES
```

### Model Candidate
```python
@dataclass  
class ModelCandidate:
    provider: str
    model: str
    score: float
    estimated_cost: float
    estimated_latency: float
    quality_match: float
    constraint_violations: List[str]
    
    def __post_init__(self):
        # Validation for TDD
        assert self.score >= 0.0
        assert self.estimated_cost >= 0.0
        assert self.estimated_latency >= 0.0
```

### Routing Decision
```python
@dataclass
class RoutingDecision:
    selected_model: ModelCandidate
    classification: PromptClassification
    alternatives: List[ModelCandidate]
    routing_time_ms: float
    confidence: float
    
    def __post_init__(self):
        # Validation for TDD
        assert 0.0 <= self.confidence <= 1.0
        assert self.routing_time_ms >= 0.0
```

## TDD Implementation Strategy

### Phase 1: Core Infrastructure (TDD)
1. **Write Tests**: Provider registry data model tests
2. **Implement**: Basic provider registry with validation
3. **Write Tests**: Scoring engine calculation tests
4. **Implement**: Scoring function with edge cases
5. **Write Tests**: Constraint validation tests
6. **Implement**: Hard/soft constraint logic

### Phase 2: Semantic Classification (TDD)
1. **Write Tests**: Embedding generation tests (mocked)
2. **Implement**: Embedding service interface
3. **Write Tests**: Vector similarity search tests
4. **Implement**: Vector store operations
5. **Write Tests**: Confidence scoring tests
6. **Implement**: Classification confidence logic

### Phase 3: LLM-Assisted Fallback (TDD)
1. **Write Tests**: LLM classification prompt tests
2. **Implement**: Classification prompt templates
3. **Write Tests**: Confidence threshold tests
4. **Implement**: Threshold management logic
5. **Write Tests**: Fallback decision tests
6. **Implement**: Complete fallback pipeline

### Phase 4: Integration & E2E (TDD)
1. **Write Tests**: Complete routing workflow tests
2. **Implement**: Router orchestration logic
3. **Write Tests**: API endpoint tests
4. **Implement**: FastAPI endpoints
5. **Write Tests**: Performance benchmark tests
6. **Implement**: Performance optimizations

## Technology Stack

### Core
- **Language**: Python 3.11+
- **Framework**: FastAPI for API server
- **Database**: SQLite for development, PostgreSQL for production
- **Vector Store**: ChromaDB for development, Pinecone for production

### Testing
- **Unit Testing**: pytest, pytest-asyncio
- **Mocking**: unittest.mock, pytest-mock
- **Property Testing**: hypothesis
- **Test Data**: factory-boy, faker
- **Coverage**: pytest-cov
- **Load Testing**: locust
- **Containers**: testcontainers-python

### ML/AI
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM Fallback**: OpenAI GPT-3.5-turbo or Anthropic Claude Haiku
- **Vector Operations**: numpy, scikit-learn

### Infrastructure  
- **Async**: asyncio, aiohttp
- **Configuration**: pydantic-settings
- **Monitoring**: prometheus, structlog

## Testing Standards

### Test Quality Criteria
1. **Fast**: Unit tests < 10ms, integration tests < 100ms
2. **Reliable**: No flaky tests, deterministic outcomes
3. **Isolated**: Tests don't depend on each other
4. **Readable**: Clear test names and assertions
5. **Maintainable**: DRY principles, shared fixtures

### Coverage Requirements
- **Minimum Coverage**: 90% line coverage
- **Critical Paths**: 100% coverage for scoring and routing logic
- **Edge Cases**: Comprehensive error handling coverage

### Test Naming Convention
```python
def test_should_[expected_behavior]_when_[condition]():
    # Arrange
    # Act  
    # Assert
    pass

# Examples:
def test_should_return_highest_scored_model_when_multiple_candidates_available():
def test_should_fallback_to_llm_when_semantic_confidence_below_threshold():
def test_should_raise_no_models_error_when_all_models_violate_constraints():
```

## Success Metrics (with Testing)

1. **Accuracy**: Classification accuracy > 90% (verified through test suite)
2. **Latency**: Routing decision < 100ms (performance tests)
3. **Cost Optimization**: 20-30% cost reduction (integration tests with mock pricing)
4. **Quality Maintenance**: Task success rate (end-to-end tests)
5. **Reliability**: 99.9% uptime (load tests and error handling tests)
6. **Test Coverage**: >90% line coverage, 100% critical path coverage

## TDD Benefits for This Project

1. **Design Clarity**: Tests force clear interface design
2. **Regression Prevention**: Catch breaking changes early
3. **Documentation**: Tests serve as living documentation
4. **Confidence**: Safe refactoring with comprehensive test coverage
5. **Quality**: Better error handling and edge case coverage
