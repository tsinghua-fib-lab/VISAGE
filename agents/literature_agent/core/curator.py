import json
import requests
import time
from typing import List, Dict
import xml.etree.ElementTree as ET
from Bio import Entrez
from Bio import Medline
import logging

class CuratorAgent:
    """
    Research librarian agent for retrieving and screening academic literature.
    """
    
    def __init__(self, model_client=None, logger=None):
        """
        Initialize Curator Agent.
        
        Args:
            model_client: LLM client instance (OpenAI, Anthropic, etc.)
            logger: Optional logger instance
        """
        self.model_client = model_client
        self.logger = logger or logging.getLogger(__name__)
        self._setup_pubmed_config("     ")  # Set your email here
        
        # Boolean search templates with synonym expansion
        self.search_templates = {
            "A1": '("built environment" OR "urban form" OR streetscape OR "physical environment" OR "neighborhood characteristics") AND ("exposure segregation" OR "socioeconomic segregation" OR "income mixing" OR "social mixing" OR "residential segregation" OR "spatial segregation") AND (urban OR neighborhood OR community OR city)',
            "A2": '("street view" OR "street-level imagery" OR satellite OR "remote sensing" OR "overhead imagery" OR "aerial imagery" OR "google street view") AND (segregation OR "social mixing" OR "exposure segregation" OR "income segregation") AND (urban OR neighborhood OR community)',
            "A3": '("exposure segregation" OR "intergroup encounters" OR "social exposure" OR "cross-group contact") AND (urban OR city OR neighborhood OR "built environment")',
            "A4": '("vision-language model" OR VLM OR multimodal OR "computer vision") AND (reasoning OR "chain of thought" OR codebook OR agent) AND (urban OR geography OR social OR "spatial analysis")'
        }
    
    def run_boolean_search(self, template_key: str, use_mock_data: bool = True) -> List[Dict]:
        """
        Execute boolean search for a specific template.
        
        Args:
            template_key: One of "A1", "A2", "A3", "A4"
            
        Returns:
            List of search results
        """
        if template_key not in self.search_templates:
            raise ValueError(f"Invalid template key: {template_key}")
        
        query = self.search_templates[template_key]
        self.logger.info(f"Executing search for {template_key}: {query}")
        
        # Use academic APIs for real search
        papers = []
        
        # Try multiple APIs in sequence
        apis_attempted = 0
        apis_successful = 0
        
        # 1. PubMed API 
        try:
            self.logger.info("Searching PubMed...")
            pubmed_papers = self._search_pubmed(query, max_results=50)
            papers.extend(pubmed_papers)
            apis_attempted += 1
            apis_successful += 1
            self.logger.info(f"Found {len(pubmed_papers)} papers from PubMed")
        except Exception as e:
            self.logger.warning(f"PubMed search failed: {e}")
        
        # 2. Semantic Scholar API
        try:
            self.logger.info("Searching Semantic Scholar...")
            semantic_papers = self._search_semantic_scholar(query, limit=50)
            papers.extend(semantic_papers)
            apis_attempted += 1
            apis_successful += 1
            self.logger.info(f"Found {len(semantic_papers)} papers from Semantic Scholar")
        except Exception as e:
            self.logger.warning(f"Semantic Scholar search failed: {e}")
        
        # 3. arXiv API (for all templates, not just A4)
        try:
            self.logger.info("Searching arXiv...")
            arxiv_papers = self._search_arxiv(query, max_results=30)
            papers.extend(arxiv_papers)
            apis_attempted += 1
            apis_successful += 1
            self.logger.info(f"Found {len(arxiv_papers)} papers from arXiv")
        except Exception as e:
            self.logger.warning(f"arXiv search failed: {e}")
        
        # 4. CrossRef API
        try:
            self.logger.info("Searching CrossRef...")
            crossref_papers = self._search_crossref(query, rows=50)
            papers.extend(crossref_papers)
            apis_attempted += 1
            apis_successful += 1
            self.logger.info(f"Found {len(crossref_papers)} papers from CrossRef")
        except Exception as e:
            self.logger.warning(f"CrossRef search failed: {e}")
        
        # 5. OpenAlex API
        try:
            self.logger.info("Searching OpenAlex...")
            openalex_papers = self._search_openalex(query, per_page=50)
            papers.extend(openalex_papers)
            apis_attempted += 1
            apis_successful += 1
            self.logger.info(f"Found {len(openalex_papers)} papers from OpenAlex")
        except Exception as e:
            self.logger.warning(f"OpenAlex search failed: {e}")
        
        # Always include mock data papers if enabled
        if use_mock_data:
            self.logger.info("Including mock data papers...")
            mock_papers = self._get_extended_mock_papers(template_key)
            papers.extend(mock_papers)
            self.logger.info(f"Added {len(mock_papers)} mock papers")
        
        # Deduplicate papers
        unique_papers = self._deduplicate_papers(papers)
        self.logger.info(f"After deduplication: {len(unique_papers)} unique papers")
        
        # Log final statistics
        api_papers_count = len([p for p in unique_papers if p.get('source') != 'mock'])
        mock_papers_count = len([p for p in unique_papers if p.get('source') == 'mock'])
        
        self.logger.info(f"Total papers found: {len(unique_papers)} "
                        f"({api_papers_count} from APIs, {mock_papers_count} mock data) "
                        f"from {apis_successful}/{apis_attempted} APIs")
        
        return unique_papers

    def _search_pubmed(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        Search PubMed using E-utilities API.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of paper dictionaries
        """
        papers = []
        
        try:
            # Method 1: Using Biopython (recommended)
            papers = self._search_pubmed_biopython(query, max_results)
        except ImportError:
            # Method 2: Using direct API calls if Biopython not available
            self.logger.warning("Biopython not available, using direct API calls")
            papers = self._search_pubmed_direct(query, max_results)
        
        return papers

    def _search_pubmed_biopython(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        Search PubMed using Biopython library.
        
        Requires: pip install biopython
        """
        # Set email (required by NCBI)
        Entrez.email = self.pubmed_email if hasattr(self, 'pubmed_email') else "your_email@example.com"
        
        # Search PubMed
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        
        pmids = record["IdList"]
        
        if not pmids:
            return []
        
        # Fetch details for the papers
        handle = Entrez.efetch(db="pubmed", id=pmids, rettype="medline", retmode="text")
        records = list(Medline.parse(handle))
        handle.close()
        
        papers = []
        for record in records:
            paper = {
                'title': record.get('TI', 'No title'),
                'abstract': record.get('AB', ''),
                'authors': record.get('AU', []),
                'journal': record.get('JT', ''),
                'year': record.get('DP', '').split(' ')[0] if record.get('DP') else '',
                'doi': record.get('AID', [''])[0] if record.get('AID') else '',
                'pubmed_id': record.get('PMID', ''),
                'source': 'pubmed',
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{record.get('PMID', '')}",
                'query': query
            }
            
            # Clean up data
            if paper['doi'] and '[doi]' in paper['doi']:
                paper['doi'] = paper['doi'].replace('[doi]', '').strip()
            
            papers.append(paper)
        
        # Rate limiting
        time.sleep(0.5)
        
        return papers

    def _search_pubmed_direct(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        Search PubMed using direct API calls without Biopython.
        """
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        papers = []
        
        # Search endpoint
        search_url = f"{base_url}esearch.fcgi"
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }
        
        response = requests.get(search_url, params=search_params, timeout=30)
        response.raise_for_status()
        
        search_data = response.json()
        pmids = search_data.get('esearchresult', {}).get('idlist', [])
        
        if not pmids:
            return papers
        
        # Fetch details endpoint
        fetch_url = f"{base_url}efetch.fcgi"
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml'
        }
        
        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=30)
        fetch_response.raise_for_status()
        
        # Parse XML response
        papers = self._parse_pubmed_xml(fetch_response.text, query)
        
        # Rate limiting
        time.sleep(0.5)
        
        return papers

    def _parse_pubmed_xml(self, xml_content: str, query: str) -> List[Dict]:
        """
        Parse PubMed XML response.
        """
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article in root.findall('.//PubmedArticle'):
                # Extract basic information
                pmid_elem = article.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else ''
                
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None else 'No title'
                
                # Extract abstract
                abstract_text = ''
                abstract_elems = article.findall('.//AbstractText')
                for elem in abstract_elems:
                    if elem.text:
                        abstract_text += elem.text + ' '
                abstract_text = abstract_text.strip()
                
                # Extract authors
                authors = []
                for author_elem in article.findall('.//Author'):
                    last_name = author_elem.find('LastName')
                    fore_name = author_elem.find('ForeName')
                    if last_name is not None and fore_name is not None:
                        authors.append(f"{fore_name.text} {last_name.text}")
                
                # Extract journal info
                journal_elem = article.find('.//Journal/Title')
                journal = journal_elem.text if journal_elem is not None else ''
                
                # Extract publication date
                year_elem = article.find('.//PubDate/Year')
                year = year_elem.text if year_elem is not None else ''
                
                # Extract DOI
                doi = ''
                article_id_elems = article.findall('.//ArticleId')
                for id_elem in article_id_elems:
                    if id_elem.get('IdType') == 'doi':
                        doi = id_elem.text
                        break
                
                paper = {
                    'title': title,
                    'abstract': abstract_text,
                    'authors': authors,
                    'journal': journal,
                    'year': year,
                    'doi': doi,
                    'pubmed_id': pmid,
                    'source': 'pubmed',
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}",
                    'query': query
                }
                
                papers.append(paper)
                
        except ET.ParseError as e:
            self.logger.error(f"Error parsing PubMed XML: {e}")
        
        return papers

    def _setup_pubmed_config(self, email: str = None):
        """
        Setup PubMed configuration.
        
        Args:
            email: Email address for NCBI API (required)
        """
        if email:
            self.pubmed_email = email
        elif not hasattr(self, 'pubmed_email'):
            # Set a default email or raise error
            self.pubmed_email = "your_email@example.com"
            self.logger.warning("Using default PubMed email. Please set your email for better API access.")
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Fast deduplication based on title"""
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            title = paper.get('title', '').strip().lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)
        
        return unique_papers

    def _search_crossref(self, query: str, rows: int = 100) -> List[Dict]:
        """Enhanced CrossRef search"""
        import requests
        import time
        
        url = "https://api.crossref.org/works"
        params = {
            'query': query,
            'rows': rows,
            'sort': 'relevance',
            'order': 'desc',
            'filter': 'type:journal-article,from-pub-date:2010'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            papers = []
            for item in response.json().get('message', {}).get('items', []):
                # Extract author information
                authors = []
                for author in item.get('author', []):
                    given = author.get('given', '')
                    family = author.get('family', '')
                    if given and family:
                        authors.append(f"{given} {family}")
                    elif family:
                        authors.append(family)
                
                paper = {
                    'doi': item.get('DOI'),
                    'title': item.get('title', [''])[0] if item.get('title') else 'No title',
                    'authors': authors,
                    'year': int(item.get('created', {}).get('date-parts', [[0]])[0][0]) if item.get('created') else None,
                    'venue': item.get('container-title', [''])[0] if item.get('container-title') else '',
                    'url': f"https://doi.org/{item.get('DOI', '')}",
                    'abstract': item.get('abstract', ''),
                    'citations': item.get('is-referenced-by-count', 0),
                    'topics': []
                }
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            self.logger.error(f"CrossRef search error: {e}")
            return []

    def _search_openalex(self, query: str, per_page: int = 100) -> List[Dict]:
        """Enhanced OpenAlex search"""
        import requests
        import time
        
        url = "https://api.openalex.org/works"
        params = {
            'search': query,
            'per-page': per_page,
            'sort': 'cited_by_count:desc',
            'filter': 'publication_year:>2010,type:journal-article'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            papers = []
            for item in response.json().get('results', []):
                # Process author information
                authors = [authorship['author']['display_name'] 
                        for authorship in item.get('authorships', [])]
                
                # Process abstract
                abstract = ""
                if item.get('abstract_inverted_index'):
                    try:
                        abstract_words = []
                        for word, positions in item['abstract_inverted_index'].items():
                            for pos in positions:
                                abstract_words.append((pos, word))
                        abstract_words.sort()
                        abstract = ' '.join([word for pos, word in abstract_words])
                    except:
                        abstract = ""
                
                paper = {
                    'doi': item.get('doi', '').replace('https://doi.org/', ''),
                    'title': item.get('title', ''),
                    'authors': authors,
                    'year': item.get('publication_year'),
                    'venue': item.get('primary_location', {}).get('source', {}).get('display_name', ''),
                    'url': item.get('doi', ''),
                    'abstract': abstract,
                    'citations': item.get('cited_by_count', 0),
                    'topics': [topic['display_name'] for topic in item.get('topics', [])]
                }
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            self.logger.error(f"OpenAlex search error: {e}")
            return []

    def _search_semantic_scholar(self, query: str, limit: int = 50) -> List[Dict]:
        """Search using Semantic Scholar API with rate limiting"""
        import requests
        import time
        
        time.sleep(0.5)  # Basic rate limiting
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': query,
            'limit': limit,
            'fields': 'paperId,title,authors,year,venue,abstract,url,citationCount'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            # Check if rate limit is reached
            if response.status_code == 429:
                self.logger.warning("Semantic Scholar rate limit reached, waiting 10 seconds...")
                time.sleep(10)  # Wait 10 seconds before retry
                response = requests.get(url, params=params, timeout=30)
            
            response.raise_for_status()
            
            papers = []
            for item in response.json().get('data', []):
                paper = {
                    'doi': item.get('paperId'),
                    'title': item.get('title', ''),
                    'authors': [author.get('name') for author in item.get('authors', [])],
                    'year': item.get('year'),
                    'venue': item.get('venue', ''),
                    'url': item.get('url', ''),
                    'abstract': item.get('abstract', ''),
                    'citations': item.get('citationCount', 0),
                    'topics': []
                }
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            self.logger.warning(f"Semantic Scholar search failed: {e}")
            return []
    
    def _search_arxiv(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search using arXiv API"""
        import arxiv
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in client.results(search):
            paper = {
                'doi': result.entry_id,
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'year': result.published.year,
                'venue': 'arXiv',
                'url': result.entry_id,
                'abstract': result.summary,
                'citations': 0,  # arXiv doesn't have citation data
                'topics': [str(cat) for cat in result.categories]
            }
            papers.append(paper)
        
        return papers

    def _get_extended_mock_papers(self, template_key: str) -> List[Dict]:
        """Generate mock paper data for testing"""
        base_papers = [
            {
                'doi': '10.1234/urban1',
                'title': 'The Impact of Built Environment on Socioeconomic Segregation in Metropolitan Areas',
                'authors': ['Smith, J.', 'Johnson, A.', 'Lee, K.'],
                'year': 2023,
                'venue': 'Urban Studies',
                'url': 'https://doi.org/10.1234/urban1',
                'abstract': 'This study examines how physical urban form characteristics including building density, green space distribution, and transportation infrastructure influence patterns of socioeconomic exposure segregation. Using street view imagery and satellite data, we identify observable cues that correlate with segregation metrics.',
                'citations': 45,
                'topics': ['built environment', 'segregation', 'urban form'],
                'full_text': 'Full text discussing building height, green spaces, and road networks...'
            },
            {
                'doi': '10.5678/imagery2',
                'title': 'Street View Imagery Analysis for Measuring Social Mixing in Diverse Neighborhoods',
                'authors': ['Chen, W.', 'Garcia, M.', 'Patel, R.'],
                'year': 2022,
                'venue': 'Computers, Environment and Urban Systems',
                'url': 'https://doi.org/10.5678/imagery2',
                'abstract': 'We demonstrate how Google Street View imagery can be used to detect visual cues of social mixing, including commercial diversity, public space quality, and housing characteristics. Machine learning models identify features predictive of income diversity.',
                'citations': 32,
                'topics': ['street view', 'social mixing', 'computer vision'],
                'full_text': 'Analysis of storefronts, sidewalk conditions, and building maintenance...'
            },
            {
                'doi': '10.9012/satellite3',
                'title': 'Satellite-Based Assessment of Urban Form and Exposure Segregation',
                'authors': ['Brown, T.', 'Davis, L.', 'Wilson, P.'],
                'year': 2023,
                'venue': 'Remote Sensing',
                'url': 'https://doi.org/10.9012/satellite3',
                'abstract': 'Using high-resolution satellite imagery, this research maps urban morphology features such as parcel sizes, road networks, and green space distribution to analyze their relationship with socioeconomic exposure patterns across 50 cities.',
                'citations': 28,
                'topics': ['remote sensing', 'urban morphology', 'segregation'],
                'full_text': 'Examination of roof types, parking lot sizes, and vegetation coverage...'
            },
            {
                'doi': '10.3456/mixing4',
                'title': 'Intergroup Encounters and Social Exposure in Urban Public Spaces',
                'authors': ['Wilson, R.', 'Zhang, L.', 'Martinez, P.'],
                'year': 2021,
                'venue': 'Social Science Research',
                'url': 'https://doi.org/10.3456/mixing4',
                'abstract': 'This research investigates how the design and accessibility of public spaces facilitate or hinder intergroup encounters and social exposure across socioeconomic boundaries in diverse urban contexts.',
                'citations': 18,
                'topics': ['social exposure', 'public space', 'intergroup encounters'],
                'full_text': 'Study of parks, plazas, and other gathering spaces...'
            }
                ]
        
        # Filter papers based on template type
        filtered_papers = []
        for paper in base_papers:
            if self._paper_matches_template(paper, template_key):
                filtered_papers.append(paper)
        
        return filtered_papers

    def _paper_matches_template(self, paper: Dict, template_key: str) -> bool:
        """Check if paper matches search template criteria"""
        abstract = paper.get('abstract', '').lower()
        title = paper.get('title', '').lower()
        
        template_keywords = {
            'A1': ['built environment', 'urban form', 'segregation', 'mixing'],
            'A2': ['street view', 'satellite', 'imagery', 'remote sensing'],
            'A3': ['exposure segregation', 'intergroup encounters', 'social exposure'],
            'A4': ['vision-language', 'multimodal', 'vlm', 'computer vision']
        }
        
        keywords = template_keywords.get(template_key, [])
        return any(keyword in abstract + title for keyword in keywords)

    def screen_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Screen papers based on inclusion/exclusion criteria.
        
        Args:
            papers: List of paper metadata
            
        Returns:
            Filtered list of papers
        """
        screened = []
        
        for paper in papers:
            # Apply screening criteria
            if self._meets_screening_criteria(paper):
                screened.append(paper)
        
        return screened
    
    def _meets_screening_criteria(self, paper: Dict) -> bool:
        """
        Check if paper meets inclusion criteria.
        """
        # Empirical studies connecting built environment to segregation/mixing
        # Features detectable in imagery
        # Exclude mobility-only studies
        abstract = paper.get('abstract', '').lower()
        title = paper.get('title', '').lower()
        
        # Positive indicators
        positive_terms = ['built environment', 'urban form', 'street view', 'satellite', 
                         'imagery', 'visual', 'observable', 'physical environment']
        
        # Negative indicators (exclusion)
        negative_terms = ['mobility only', 'travel behavior', 'commuting', 'transportation']
        
        has_positive = any(term in abstract + title for term in positive_terms)
        has_negative = any(term in abstract + title for term in negative_terms)
        
        return has_positive and not has_negative
    
    def deduplicate_and_rank(self, papers: List[Dict]) -> List[Dict]:
        """
        Deduplicate papers and rank by relevance.
        
        Args:
            papers: List of paper metadata
            
        Returns:
            Deduplicated and ranked papers
        """
        # Deduplicate by DOI/Title
        seen_dois = set()
        seen_titles = set()
        deduped = []
        
        for paper in papers:
            doi = paper.get('doi')
            title = paper.get('title', '').strip().lower()
            
            if doi and doi in seen_dois:
                continue
            if title in seen_titles:
                continue
                
            seen_dois.add(doi)
            seen_titles.add(title)
            deduped.append(paper)

        # Ensure citations and years are integers for sorting
        for paper in deduped:
            # Convert citations to integer
            paper['citations'] = self._safe_int_conversion(paper.get('citations'), 0)
            
            # Convert year to integer with robust parsing
            paper['year'] = self._parse_year(paper.get('year'))

        # Rank by relevance, recency, citations
        deduped.sort(key=lambda x: (
            -x.get('citations', 0),  # Higher citations first
            -x.get('year', 0),       # More recent first
            self._calculate_relevance_score(x)
        ))
        
        return deduped

    def _safe_int_conversion(self, value, default=0):
        """Safely convert value to integer."""
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def _parse_year(self, year_value):
        """Parse year from various formats."""
        if year_value is None:
            return 0
        
        try:
            year_str = str(year_value).strip()
            if not year_str:
                return 0
            
            # Handle common year formats:
            # "2023", "2023-10", "2023/12", "2023-10-15", "2023a", etc.
            import re
            
            # Look for 4-digit years
            year_match = re.search(r'\b(19|20)\d{2}\b', year_str)
            if year_match:
                return int(year_match.group())
            
            # If no 4-digit year found, try to extract any number that could be a year
            numbers = re.findall(r'\d+', year_str)
            for num in numbers:
                if len(num) == 4 and 1900 <= int(num) <= 2100:
                    return int(num)
            
            return 0
            
        except (ValueError, TypeError):
            return 0
    
    def _calculate_relevance_score(self, paper: Dict) -> float:
        """Calculate relevance score for ranking."""
        score = 0
        abstract = paper.get('abstract', '').lower()
        title = paper.get('title', '').lower()
        
        # High relevance terms
        high_relevance = ['exposure segregation', 'street view', 'satellite imagery', 
                         'built environment', 'visual cues']
        for term in high_relevance:
            if term in abstract or term in title:
                score += 2
        
        # Medium relevance terms  
        med_relevance = ['segregation', 'social mixing', 'urban form', 'neighborhood']
        for term in med_relevance:
            if term in abstract or term in title:
                score += 1
                
        return score
    
    def _truncate_abstract(self, abstract: str, max_words: int = 150) -> str:
        """Truncate abstract to specified word count."""
        words = abstract.split()
        if len(words) <= max_words:
            return abstract
        return ' '.join(words[:max_words]) + '...'
    
    def _generate_keep_reason(self, paper: Dict) -> str:
        """Generate reason for keeping the paper."""
        reasons = []
        abstract = paper.get('abstract', '').lower()
        
        if any(term in abstract for term in ['street view', 'street-level']):
            reasons.append("street-view imagery")
        if any(term in abstract for term in ['satellite', 'remote sensing', 'aerial']):
            reasons.append("satellite imagery") 
        if 'built environment' in abstract:
            reasons.append("built environment focus")
        if 'segregation' in abstract or 'mixing' in abstract:
            reasons.append("segregation/mixing focus")
            
        return f"Empirical study with {', '.join(reasons)}" if reasons else "Relevant urban study"
    
    def _infer_perspective(self, paper: Dict) -> str:
        """Infer imagery perspective from paper content."""
        abstract = paper.get('abstract', '').lower()
        title = paper.get('title', '').lower()
        
        has_street_view = any(term in abstract + title for term in ['street view', 'street-level'])
        has_remote = any(term in abstract + title for term in ['satellite', 'remote sensing', 'aerial'])
        
        if has_street_view and has_remote:
            return "both"
        elif has_street_view:
            return "street_view" 
        elif has_remote:
            return "remote_sensing"
        else:
            return "unclear"
    
    def _extract_visual_cue_hint(self, paper: Dict) -> str:
        """Extract visual cue hints from paper."""
        abstract = paper.get('abstract', '')
        # Simple keyword extraction - could be enhanced with NLP
        cues = []
        
        cue_keywords = {
            'building': ['building', 'architecture', 'facade'],
            'green': ['green', 'tree', 'vegetation', 'park'],
            'road': ['road', 'street', 'highway', 'intersection'],
            'commercial': ['commercial', 'store', 'business', 'retail'],
            'housing': ['housing', 'residential', 'dwelling']
        }
        
        for cue_type, keywords in cue_keywords.items():
            if any(keyword in abstract.lower() for keyword in keywords):
                cues.append(cue_type)
        
        return ', '.join(cues[:3]) if cues else "built environment features"
    
    def execute_workflow(self) -> Dict:
        """
        Execute complete curator workflow.
        
        Returns:
            Dictionary with both PRISMA counts and curated items
        """
        self.logger.info("Starting Curator workflow")
        
        # Step 1: Run all boolean searches
        all_found = []
        for template_key in self.search_templates.keys():
            results = self.run_boolean_search(template_key)
            all_found.extend(results)
        
        # Step 2: Deduplicate and rank
        deduped = self.deduplicate_and_rank(all_found)
        
        # Step 3: Screen papers
        screened = self.screen_papers(deduped)
        
        # Step 4: Final inclusion
        included = screened
        
        # Generate output in single dictionary format for backward compatibility
        output = {
            "prisma_counts": {
                "found": len(all_found),
                "deduped": len(deduped), 
                "screened": len(screened),
                "included": len(included)
            },
            "curator_table": [
                {
                    "DOI": paper.get('doi'),
                    "Title": paper.get('title', ''),
                    "Authors": paper.get('authors', []),
                    "Year": paper.get('year'),
                    "Venue": paper.get('venue'),
                    "URL": paper.get('url'),
                    "Abstract150w": self._truncate_abstract(paper.get('abstract', '')),
                    "TopicTags": paper.get('topics', []),
                    "KeepReason": self._generate_keep_reason(paper),
                    "PerspectiveHint": self._infer_perspective(paper),
                    "VisualCueHint": self._extract_visual_cue_hint(paper),
                    "Accepted": True
                }
                for paper in included
            ]
        }
        
        self.logger.info(f"Curator workflow completed: {output['prisma_counts']}")
        return output

# Example usage with file output
if __name__ == "__main__":
    # Initialize curator agent
    curator = CuratorAgent()
    
    # Execute workflow and get single dictionary result
    result = curator.execute_workflow()
    
    # Save PRISMA counts to file
    with open('/data3/maruolong/VISAGE/data/prisma_counts.json', 'w', encoding='utf-8') as f:
        json.dump({"prisma_counts": result["prisma_counts"]}, f, indent=2, ensure_ascii=False)
    
    # Save curated items to file
    with open('/data3/maruolong/VISAGE/data/curator_table.json', 'w', encoding='utf-8') as f:
        json.dump({"curator_table": result["curator_table"]}, f, indent=2, ensure_ascii=False)
    
    print("✅ PRISMA counts saved to: /data3/maruolong/VISAGE/data/prisma_counts.json")
    print("✅ Curated items saved to: /data3/maruolong/VISAGE/data/curator_table.json")
    
    # Also print to console for verification
    print("\n📊 PRISMA Counts:")
    print(json.dumps(result["prisma_counts"], indent=2))
    
    print(f"\n📚 Curated Items ({len(result['curator_table'])} papers):")
    for i, paper in enumerate(result['curator_table']):
        print(f"  {i+1}. {paper['Title']}")
        print(f"     Authors: {', '.join(paper['Authors'][:2])}...")
        print(f"     Year: {paper['Year']}, Perspective: {paper['PerspectiveHint']}")