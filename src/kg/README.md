# Knowledge Graph Enhanced RAG

Extended medical RAG system with comprehensive medical datasets and knowledge graph integration.

## 🎯 Features

- **PubMed Integration**: 5,000+ peer-reviewed medical abstracts
- **Clinical Documentation**: MTSamples medical transcriptions 
- **Medical Terminology**: MeSH vocabulary and concepts
- **Evidence Stratification**: Quality-weighted retrieval
- **Temporal Awareness**: Publication date consideration

## 📁 Data Sources

### PubMed Abstracts
- **Size**: ~5,000 abstracts across 10 medical topics
- **Quality**: Peer-reviewed research articles
- **Coverage**: Diabetes, hypertension, cardiology, etc.
- **Metadata**: Authors, publication year, journal

### MTSamples Transcriptions  
- **Size**: ~100 clinical documentation samples
- **Types**: Consultation notes, procedure reports
- **Specialties**: Cardiology, endocrinology, obstetrics
- **Metadata**: Medical specialty, document type

### MeSH Vocabulary
- **Size**: ~50 core medical concepts
- **Structure**: Hierarchical medical terminology
- **Coverage**: Diseases, procedures, phenomena
- **Metadata**: Synonyms, tree numbers, definitions

## 🚀 Quick Start

### 1. Fetch Medical Datasets

```bash
# Fetch all datasets (recommended)
python fetch_data.py --source all --max-results 500

# Fetch specific source
python fetch_data.py --source pubmed --max-results 1000
python fetch_data.py --source mtsamples
python fetch_data.py --source mesh
```

### 2. Check Downloaded Data

```bash
# View data structure
ls data/kg_raw/
# pubmed/    mtsamples/    mesh/    combined/

# Check statistics
cat data/kg_raw/combined/dataset_statistics.json
```

### 3. Data Organization

```
data/kg_raw/
├── pubmed/
│   ├── pubmed_complete.json      # All PubMed abstracts
│   └── pubmed_metadata.json      # Dataset statistics
├── mtsamples/
│   ├── mtsamples_complete.json   # All MTSamples docs
│   └── mtsamples_metadata.json   # Dataset statistics  
├── mesh/
│   ├── mesh_complete.json        # All MeSH concepts
│   └── mesh_metadata.json       # Dataset statistics
└── combined/
    ├── all_medical_data.json     # Combined dataset
    └── dataset_statistics.json   # Overall statistics
```

## 📊 Expected Dataset Sizes

- **PubMed**: ~5,000 abstracts (500 per topic × 10 topics)
- **MTSamples**: ~100 clinical documents
- **MeSH**: ~50 medical concepts
- **Total**: ~5,150 medical documents
- **Processing Time**: 15-30 minutes (depending on network)

## 🔧 Configuration

The fetcher respects NCBI rate limits:
- **Rate**: 3 requests/second (without API key)
- **Batch Size**: 200 abstracts per request
- **Timeout**: 60 seconds per request
- **Retry**: Basic error handling

## 📈 Next Steps

After fetching data:

1. **Process Documents**: Chunk and prepare for vector storage
2. **Build Knowledge Graph**: Extract medical relationships  
3. **Implement Enhanced RAG**: Evidence stratification + temporal awareness
4. **Evaluate Performance**: Compare against simple RAG baseline

## 🚨 Important Notes

- **PubMed API**: No API key required, but rate-limited
- **Data Size**: ~5,150 documents vs 3 in simple version
- **Processing**: Significantly more comprehensive than test data
- **Quality**: Real medical literature vs sample text files

## 🔍 Data Quality

- **High Quality**: PubMed (peer-reviewed research)
- **Medium Quality**: MTSamples (clinical documentation)  
- **Authoritative**: MeSH (NLM standard terminology)
- **Diverse**: Research + clinical + terminology coverage