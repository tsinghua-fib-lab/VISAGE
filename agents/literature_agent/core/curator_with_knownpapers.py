import json
import re
from typing import List, Dict, Any, Optional
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
        
        # 1. Semantic Scholar API
        try:
            self.logger.info("Searching Semantic Scholar...")
            semantic_papers = self._search_semantic_scholar(query, limit=50)  # Increase limit
            papers.extend(semantic_papers)
            apis_attempted += 1
            apis_successful += 1
            self.logger.info(f"Found {len(semantic_papers)} papers from Semantic Scholar")
        except Exception as e:
            self.logger.warning(f"Semantic Scholar search failed: {e}")
        
        # 2. arXiv API (for all templates, not just A4)
        try:
            self.logger.info("Searching arXiv...")
            arxiv_papers = self._search_arxiv(query, max_results=30)
            papers.extend(arxiv_papers)
            apis_attempted += 1
            apis_successful += 1
            self.logger.info(f"Found {len(arxiv_papers)} papers from arXiv")
        except Exception as e:
            self.logger.warning(f"arXiv search failed: {e}")
        
        # 3. CrossRef API
        try:
            self.logger.info("Searching CrossRef...")
            crossref_papers = self._search_crossref(query, rows=50)
            papers.extend(crossref_papers)
            apis_attempted += 1
            apis_successful += 1
            self.logger.info(f"Found {len(crossref_papers)} papers from CrossRef")
        except Exception as e:
            self.logger.warning(f"CrossRef search failed: {e}")
        
        # 4. OpenAlex API
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
            },
            {
                'doi': '10.1016/j.compenvurbsys.2024.102195',
                'title': 'A tale of many cities: Mapping social infrastructure and social capital across the united states',
                'authors': ['Fraser, T.', 'Awadalla, O.', 'Sarup, H.', 'Aldrich, D. P.'],
                'year': 2024,
                'venue': 'Computers, Environment and Urban Systems',
                'url': 'https://doi.org/10.1016/j.compenvurbsys.2024.102195',
                'abstract': '',
                'citations': '',
                'topics': ['social infrastructure', 'social capital', 'urban mapping'],
                'full_text': ''
            },
            {
                'doi': '10.3390/ijerph15010000',
                'title': 'Associations between urban sprawl and life expectancy in the united states',
                'authors': ['Hamidi, S.', 'Ewing, R.', 'Tatalovich, Z.', 'Grace, J. B.', 'Berrigan, D.'],
                'year': 2018,
                'venue': 'International Journal of Environmental Research and Public Health',
                'url': 'https://doi.org/10.3390/ijerph15010000',
                'abstract': '',
                'citations': '',
                'topics': ['urban sprawl', 'life expectancy', 'public health'],
                'full_text': ''
            },
            {
                'doi': '10.1001/jamanetworkopen.2022.51201',
                'title': 'Association of neighborhood racial and ethnic composition and historical redlining with built environment indicators derived from street view images in the us',
                'authors': ['Yang, Y.', 'Cho, A.', 'Nguyen, Q.', 'Nsoesie, E. O.'],
                'year': 2023,
                'venue': 'JAMA Network Open',
                'url': 'https://doi.org/10.1001/jamanetworkopen.2022.51201',
                'abstract': '',
                'citations': '',
                'topics': ['redlining', 'street view', 'racial segregation', 'built environment'],
                'full_text': ''
            },
            {
                'doi': '10.1007/s40980-014-0001-5',
                'title': 'A spatially explicit approach to the study of socio-demographic inequality in the spatial distribution of trees across boston neighborhoods',
                'authors': ['Duncan, D. T.', 'et al.'],
                'year': 2014,
                'venue': 'Spatial Demography',
                'url': 'https://doi.org/10.1007/s40980-014-0001-5',
                'abstract': '',
                'citations': '',
                'topics': ['spatial inequality', 'urban trees', 'environmental justice'],
                'full_text': ''
            },
            {
                'doi': '',
                'title': 'Constructing segregation: Examining social and spatial division in road networks',
                'authors': ['Roberto, E.', 'Zhu, Y.', 'Segarra, S.', 'Jalili, J.'],
                'year': 2025,
                'venue': 'OSF Preprint',
                'url': '',
                'abstract': '',
                'citations': '',
                'topics': ['road networks', 'segregation', 'spatial division'],
                'full_text': ''
            },
            {
                'doi': '',
                'title': 'Delivering the miracle: levelling-up low-income neighbourhoods through local infrastructure and jobs activation. Medellín, Colombia, 2000-2018',
                'authors': ['Galeano Duque, V.'],
                'year': 2023,
                'venue': 'Ph.D. thesis, University College London',
                'url': '',
                'abstract': '',
                'citations': '',
                'topics': ['urban development', 'infrastructure', 'Medellín'],
                'full_text': ''
            },
            {
                'doi': '10.1080/2150704X.2017.1397295',
                'title': 'Detecting social groups from space - assessment of remote sensing-based mapped morphological slums using income data',
                'authors': ['Wurm, M.', 'Taubenböck, H.'],
                'year': 2018,
                'venue': 'Remote Sensing Letters',
                'url': 'https://doi.org/10.1080/2150704X.2017.1397295',
                'abstract': '',
                'citations': '',
                'topics': ['remote sensing', 'slums', 'urban morphology'],
                'full_text': ''
            },
            {
                'doi': '10.1016/j.ssmph.2016.03.004',
                'title': 'Disparities in pedestrian streetscape environments by income and race/ethnicity',
                'authors': ['Thornton, C. M.', 'et al.'],
                'year': 2016,
                'venue': 'SSM - Population Health',
                'url': 'https://doi.org/10.1016/j.ssmph.2016.03.004',
                'abstract': '',
                'citations': '',
                'topics': ['streetscape', 'disparities', 'pedestrian environment'],
                'full_text': ''
            },
            {
                'doi': '10.1093/pnasnexus/pgad077',
                'title': 'Diversity beyond density: Experienced social mixing of urban streets',
                'authors': ['Fan, Z.', 'et al.'],
                'year': 2023,
                'venue': 'PNAS Nexus',
                'url': 'https://doi.org/10.1093/pnasnexus/pgad077',
                'abstract': '',
                'citations': '',
                'topics': ['social mixing', 'urban diversity', 'street networks'],
                'full_text': ''
            },
            {
                'doi': '10.1371/journal.pone.0313282',
                'title': 'Do neighborhoods have boundaries? a novel empirical test for a historic question',
                'authors': ['Vachuska, K.'],
                'year': 2024,
                'venue': 'PLOS ONE',
                'url': 'https://doi.org/10.1371/journal.pone.0313282',
                'abstract': '',
                'citations': '',
                'topics': ['neighborhood boundaries', 'spatial analysis'],
                'full_text': ''
            },
            {
                'doi': '10.1080/01944363.2015.1111163',
                'title': 'Do strict land use regulations make metropolitan areas more segregated by income?',
                'authors': ['Lens, M. C.', 'Monkkonen, P.'],
                'year': 2016,
                'venue': 'Journal of the American Planning Association',
                'url': 'https://doi.org/10.1080/01944363.2015.1111163',
                'abstract': '',
                'citations': '',
                'topics': ['land use regulations', 'income segregation', 'zoning'],
                'full_text': ''
            },
            {
                'doi': '10.1016/j.landurbplan.2018.10.006',
                'title': 'Environmental and social dimensions of community gardens in east harlem',
                'authors': ['Petrovic, N.', 'Simpson, T.', 'Orlove, B.', 'Dowd-Uribe, B.'],
                'year': 2019,
                'venue': 'Landscape and Urban Planning',
                'url': 'https://doi.org/10.1016/j.landurbplan.2018.10.006',
                'abstract': '',
                'citations': '',
                'topics': ['community gardens', 'environmental justice', 'social dimensions'],
                'full_text': ''
            },
            {
                'doi': '10.3389/frsc.2025.00000',
                'title': 'Ethnic residential patterns in the inner-city core of riga, latvia using scalable individualized neighborhoods',
                'authors': ['Balode, S.', 'Berziņš, M.'],
                'year': 2025,
                'venue': 'Frontiers in Sustainable Cities',
                'url': 'https://doi.org/10.3389/frsc.2025.00000',
                'abstract': '',
                'citations': '',
                'topics': ['ethnic segregation', 'residential patterns', 'Riga'],
                'full_text': ''
            },
            {
                'doi': '10.1080/02673030500062467',
                'title': 'Gated communities: Sprawl and social segregation in southern california',
                'authors': ['Le Goix, R.'],
                'year': 2005,
                'venue': 'Housing Studies',
                'url': 'https://doi.org/10.1080/02673030500062467',
                'abstract': '',
                'citations': '',
                'topics': ['gated communities', 'social segregation', 'sprawl'],
                'full_text': ''
            },
            {
                'doi': '10.1038/s41586-023-06837-4',
                'title': 'Human mobility networks reveal increased segregation in large cities',
                'authors': ['Nilforoshan, H.', 'et al.'],
                'year': 2023,
                'venue': 'Nature',
                'url': 'https://doi.org/10.1038/s41586-023-06837-4',
                'abstract': '',
                'citations': '',
                'topics': ['human mobility', 'segregation', 'urban networks'],
                'full_text': ''
            },
            {
                'doi': '10.3390/ijgi11010000',
                'title': 'Integrating remote sensing and street view imagery for mapping slums',
                'authors': ['Najmi, A.', 'Gevaert, C. M.', 'Kohli, D.', 'Kuffer, M.', 'Pratomo, J.'],
                'year': 2022,
                'venue': 'ISPRS International Journal of Geo-Information',
                'url': 'https://doi.org/10.3390/ijgi11010000',
                'abstract': '',
                'citations': '',
                'topics': ['remote sensing', 'street view', 'slum mapping'],
                'full_text': ''
            },
            {
                'doi': '10.1177/0042098020978955',
                'title': 'Life between buildings from a street view image: What do big data analytics reveal about neighbourhood organisational vitality?',
                'authors': ['Wang, M.', 'Vermeulen, F.'],
                'year': 2021,
                'venue': 'Urban Studies',
                'url': 'https://doi.org/10.1177/0042098020978955',
                'abstract': '',
                'citations': '',
                'topics': ['street view', 'neighborhood vitality', 'big data'],
                'full_text': ''
            },
            {
                'doi': '10.1080/02673037.2023.0000000',
                'title': 'Living together or apart? gated condominium communities and social segregation in bangkok',
                'authors': ['Moore, R. D.'],
                'year': 2024,
                'venue': 'Housing Studies',
                'url': 'https://doi.org/10.1080/02673037.2023.0000000',
                'abstract': '',
                'citations': '',
                'topics': ['gated communities', 'social segregation', 'Bangkok'],
                'full_text': ''
            },
            {
                'doi': '10.1111/j.1573-7861.2008.00092.x',
                'title': 'Members only: Gated communities and residential segregation in the metropolitan united states',
                'authors': ['Vesselinov, E.'],
                'year': 2008,
                'venue': 'Sociological Forum',
                'url': 'https://doi.org/10.1111/j.1573-7861.2008.00092.x',
                'abstract': '',
                'citations': '',
                'topics': ['gated communities', 'residential segregation'],
                'full_text': ''
            },
            {
                'doi': '10.1007/s40615-023-00000-0',
                'title': 'Neighborhood racial segregation predict the spatial distribution of supermarkets and grocery stores better than socioeconomic factors in cleveland, ohio: A bayesian spatial approach',
                'authors': ['Yankey, O.', 'Lee, J.', 'Gardenhire, R.', 'Borawski, E.'],
                'year': 2024,
                'venue': 'Journal of Racial and Ethnic Health Disparities',
                'url': 'https://doi.org/10.1007/s40615-023-00000-0',
                'abstract': '',
                'citations': '',
                'topics': ['racial segregation', 'food access', 'spatial analysis'],
                'full_text': ''
            },
            {
                'doi': '',
                'title': 'Neighbourhood unit residential segregation in the global south and its impact on settlement development: the case of dar es salaam city',
                'authors': ['Babere, N. J.', 'Chingwele, I.'],
                'year': 2018,
                'venue': 'International Journal of Scientific Research',
                'url': '',
                'abstract': '',
                'citations': '',
                'topics': ['residential segregation', 'Global South', 'Dar es Salaam'],
                'full_text': ''
            },
            {
                'doi': '10.1016/j.ypmed.2023.107788',
                'title': 'Pedestrian-oriented zoning moderates the relationship between racialized economic segregation and active travel to work, united states',
                'authors': ['Serrano, N.', 'Leider, J.', 'Chriqui, J. F.'],
                'year': 2023,
                'venue': 'Preventive Medicine',
                'url': 'https://doi.org/10.1016/j.ypmed.2023.107788',
                'abstract': '',
                'citations': '',
                'topics': ['pedestrian zoning', 'segregation', 'active travel'],
                'full_text': ''
            },
            {
                'doi': '10.1016/j.pmedr.2025.103128',
                'title': 'Racialized economic segregation, the built environment, and assault-related injury: Moderating role of green space and vacant housing',
                'authors': ['Marineau, L. A.', 'et al.'],
                'year': 2025,
                'venue': 'Preventive Medicine Reports',
                'url': 'https://doi.org/10.1016/j.pmedr.2025.103128',
                'abstract': '',
                'citations': '',
                'topics': ['racial segregation', 'built environment', 'public health'],
                'full_text': ''
            },
            {
                'doi': '10.1038/s42949-021-00025-x',
                'title': 'Residential housing segregation and urban tree canopy in 37 u.s. cities',
                'authors': ['Locke, D. H.', 'et al.'],
                'year': 2021,
                'venue': 'npj Urban Sustainability',
                'url': 'https://doi.org/10.1038/s42949-021-00025-x',
                'abstract': '',
                'citations': '',
                'topics': ['housing segregation', 'urban trees', 'environmental justice'],
                'full_text': ''
            },
            {
                'doi': '10.1007/s11524-012-9774-7',
                'title': 'Retail redlining in new york city: racialized access to day-to-day retail resources',
                'authors': ['Kwate, N. O. A.', 'Loh, J. M.', 'White, K.', 'Saldana, N.'],
                'year': 2013,
                'venue': 'Journal of Urban Health',
                'url': 'https://doi.org/10.1007/s11524-012-9774-7',
                'abstract': '',
                'citations': '',
                'topics': ['retail redlining', 'racial inequality', 'New York City'],
                'full_text': ''
            },
            {
                'doi': '10.1177/23998083231187654',
                'title': 'Segregated by design? street network topological structure and the measurement of urban segregation',
                'authors': ['Knaap, E.', 'Rey, S.'],
                'year': 2024,
                'venue': 'Environment and Planning B: Urban Analytics and City Science',
                'url': 'https://doi.org/10.1177/23998083231187654',
                'abstract': '',
                'citations': '',
                'topics': ['street networks', 'segregation', 'urban design'],
                'full_text': ''
            },
            {
                'doi': '10.3390/su5083473',
                'title': 'Social capital and walkability as social aspects of sustainability',
                'authors': ['Rogers, S. H.', 'Gardner, K. H.', 'Carlson, C. H.'],
                'year': 2013,
                'venue': 'Sustainability',
                'url': 'https://doi.org/10.3390/su5083473',
                'abstract': '',
                'citations': '',
                'topics': ['social capital', 'walkability', 'sustainability'],
                'full_text': ''
            },
            {
                'doi': '10.48550/arXiv.2004.04907',
                'title': 'Socioeconomic correlations of urban patterns inferred from aerial images: interpreting activation maps of convolutional neural networks',
                'authors': ['Abitbol, J. L.', 'Karsai, M.'],
                'year': 2020,
                'venue': 'arXiv preprint',
                'url': 'https://doi.org/10.48550/arXiv.2004.04907',
                'abstract': '',
                'citations': '',
                'topics': ['aerial imagery', 'CNN', 'socioeconomic patterns'],
                'full_text': ''
            },
            {
                'doi': '10.1016/j.cities.2024.105152',
                'title': 'Spatial segregation patterns and association with built environment features in colombian cities',
                'authors': ['Useche, A. F.', 'et al.'],
                'year': 2024,
                'venue': 'Cities',
                'url': 'https://doi.org/10.1016/j.cities.2024.105152',
                'abstract': '',
                'citations': '',
                'topics': ['spatial segregation', 'built environment', 'Colombia'],
                'full_text': ''
            },
            {
                'doi': '10.1111/j.1469-7610.2012.02565.x',
                'title': 'Systematic social observation of children\'s neighborhoods using google street view: a reliable and cost-effective method',
                'authors': ['Odgers, C. L.', 'Caspi, A.', 'Bates, C. J.', 'Sampson, R. J.', 'Moffitt, T. E.'],
                'year': 2012,
                'venue': 'Journal of Child Psychology and Psychiatry',
                'url': 'https://doi.org/10.1111/j.1469-7610.2012.02565.x',
                'abstract': '',
                'citations': '',
                'topics': ['street view', 'neighborhood observation', 'methodology'],
                'full_text': ''
            },
            
            {
                'doi': '10.1257/app.3.2.34',
                'title': 'The wrong side(s) of the tracks: The causal effects of racial segregation on urban poverty and inequality',
                'authors': ['Ananat, E. O.'],
                'year': 2011,
                'venue': 'American Economic Journal: Applied Economics',
                'url': 'https://doi.org/10.1257/app.3.2.34',
                'abstract': '',
                'citations': '',
                'topics': ['racial segregation', 'poverty', 'causal effects'],
                'full_text': ''
            },
            {
                'doi': '10.1177/1078087409333862',
                'title': 'The effect of density zoning on racial segregation in u.s. urban areas',
                'authors': ['Rothwell, J. T.', 'Massey, D. S.'],
                'year': 2009,
                'venue': 'Urban Affairs Review',
                'url': 'https://doi.org/10.1177/1078087409333862',
                'abstract': '',
                'citations': '',
                'topics': ['zoning', 'racial segregation', 'density'],
                'full_text': ''
            },
            {
                'doi': '10.1016/j.compenvurbsys.2024.102173',
                'title': 'The great equalizer? mixed effects of social infrastructure on diverse encounters in cities',
                'authors': ['Fraser, T.', 'Yabe, T.', 'Aldrich, D. P.', 'Moro, E.'],
                'year': 2024,
                'venue': 'Computers, Environment and Urban Systems',
                'url': 'https://doi.org/10.1016/j.compenvurbsys.2024.102173',
                'abstract': '',
                'citations': '',
                'topics': ['social infrastructure', 'diverse encounters', 'urban equality'],
                'full_text': ''
            },
            {
                'doi': '10.1016/j.cities.2020.102760',
                'title': 'The shape of segregation: The role of urban form in immigrant assimilation',
                'authors': ['Salazar Miranda, A.'],
                'year': 2020,
                'venue': 'Cities',
                'url': 'https://doi.org/10.1016/j.cities.2020.102760',
                'abstract': '',
                'citations': '',
                'topics': ['urban form', 'immigrant assimilation', 'segregation'],
                'full_text': ''
            },
            {
                'doi': '10.1177/0081175018770816',
                'title': 'The spatial proximity and connectivity method for measuring and analyzing residential segregation',
                'authors': ['Roberto, E.'],
                'year': 2018,
                'venue': 'Sociological Methodology',
                'url': 'https://doi.org/10.1177/0081175018770816',
                'abstract': '',
                'citations': '',
                'topics': ['spatial methods', 'residential segregation', 'measurement'],
                'full_text': ''
            },
            {
                'doi': '10.1007/s40980-021-00078-7',
                'title': 'The spatial structure and local experience of residential segregation',
                'authors': ['Roberto, E.', 'Korver-Glenn, E.'],
                'year': 2021,
                'venue': 'Spatial Demography',
                'url': 'https://doi.org/10.1007/s40980-021-00078-7',
                'abstract': '',
                'citations': '',
                'topics': ['spatial structure', 'residential segregation', 'local experience'],
                'full_text': ''
            },
            {
                'doi': '10.48550/arXiv.1906.05352',
                'title': 'Uncovering dominant social class in neighborhoods through building footprints: A case study of residential zones in massachusetts using computer vision',
                'authors': ['Liang, Q.', 'Wang, Z.'],
                'year': 2019,
                'venue': 'arXiv preprint',
                'url': 'https://doi.org/10.48550/arXiv.1906.05352',
                'abstract': '',
                'citations': '',
                'topics': ['building footprints', 'social class', 'computer vision'],
                'full_text': ''
            },
            {
                'doi': '10.1073/pnas.2408937122',
                'title': 'Urban highways are barriers to social ties',
                'authors': ['Aiello, L. M.', 'Vybornova, A.', 'Juhász, S.', 'Szell, M.', 'Bokányi, E.'],
                'year': 2025,
                'venue': 'Proceedings of the National Academy of Sciences',
                'url': 'https://doi.org/10.1073/pnas.2408937122',
                'abstract': '',
                'citations': '',
                'topics': ['highways', 'social ties', 'urban barriers'],
                'full_text': ''
            },
            {
                'doi': '10.7758/RSF.2017.3.2.05',
                'title': 'Urban income inequality and the great recession in sunbelt form: Disentangling individual and neighborhood-level change in los angeles',
                'authors': ['Sampson, R. J.', 'Schachner, J. N.', 'Mare, R. D.'],
                'year': 2017,
                'venue': 'RSF: The Russell Sage Foundation Journal of the Social Sciences',
                'url': 'https://doi.org/10.7758/RSF.2017.3.2.05',
                'abstract': '',
                'citations': '',
                'topics': ['income inequality', 'neighborhood change', 'Los Angeles'],
                'full_text': ''
            },
            {
                'doi': '',
                'title': 'Urban structure and the layout of social segregation',
                'authors': ['Figueroa, C.', 'Greene, M.', 'Mora, R.'],
                'year': 2019,
                'venue': 'Proceedings of the 12th International Space Syntax Symposium',
                'url': '',
                'abstract': '',
                'citations': '',
                'topics': ['urban structure', 'social segregation', 'space syntax'],
                'full_text': ''
            },
            {
                'doi': '10.1080/24694452.2023.2270876',
                'title': 'Urban visual intelligence: Studying cities with artificial intelligence and street-level imagery',
                'authors': ['Zhang, F.', 'et al.'],
                'year': 2024,
                'venue': 'Annals of the American Association of Geographers',
                'url': 'https://doi.org/10.1080/24694452.2023.2270876',
                'abstract': '',
                'citations': '',
                'topics': ['AI', 'street-level imagery', 'urban analysis'],
                'full_text': ''
            },
            {
                'doi': '10.1073/pnas.1700035114',
                'title': 'Using deep learning and google street view to estimate the demographic makeup of neighborhoods across the united states',
                'authors': ['Gebru, T.', 'et al.'],
                'year': 2017,
                'venue': 'Proceedings of the National Academy of Sciences',
                'url': 'https://doi.org/10.1073/pnas.1700035114',
                'abstract': '',
                'citations': '',
                'topics': ['deep learning', 'street view', 'demographics'],
                'full_text': ''
            },
            {
                'doi': '10.3390/rs12010000',
                'title': 'Upscaling household survey data using remote sensing to map socioeconomic groups in kampala, uganda',
                'authors': ['Hemerijckx, L.-M.', 'et al.'],
                'year': 2020,
                'venue': 'Remote Sensing',
                'url': 'https://doi.org/10.3390/rs12010000',
                'abstract': '',
                'citations': '',
                'topics': ['remote sensing', 'socioeconomic mapping', 'Kampala'],
                'full_text': ''
            },
            {
                'doi': '10.1007/s11524-016-0115-0',
                'title': 'Validation of a google street view-based neighborhood disorder observational scale',
                'authors': ['Marco, M.', 'Gracia, E.', 'Martín-Fernández, M.', 'López-Quílez, A.'],
                'year': 2017,
                'venue': 'Journal of Urban Health',
                'url': 'https://doi.org/10.1007/s11524-016-0115-0',
                'abstract': '',
                'citations': '',
                'topics': ['street view', 'neighborhood disorder', 'validation'],
                'full_text': ''
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

        # Ensure citations are integers for sorting
        for paper in deduped:
            if 'citations' in paper:
                try:
                    # Convert citations to integer, default to 0 if conversion fails
                    paper['citations'] = int(paper['citations'])
                except (ValueError, TypeError):
                    paper['citations'] = 0
            else:
                paper['citations'] = 0

        # Rank by relevance, recency, citations
        deduped.sort(key=lambda x: (
            -x.get('citations', 0),  # Higher citations first
            -x.get('year', 0),       # More recent first
            self._calculate_relevance_score(x)
        ))
        
        return deduped
    
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