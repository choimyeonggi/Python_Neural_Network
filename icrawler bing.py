"""
BING CRAWLER WITH ICRAWLER PACKAGE.
"""

from icrawler.builtin import BingImageCrawler, GreedyImageCrawler

search_term = 'tsutsugamushi'
"""
bing_crawler = BingImageCrawler(downloader_threads=4,
                                storage={'root_dir': search_term + ' crawled images'})
bing_crawler.crawl(keyword=search_term, filters=None, offset=0, max_num=1000)
"""
#
search_url = 'https://bbc.com'
greedy_crawler = GreedyImageCrawler(storage={'root_dir': 'greedy_bing_eng_url ' + search_term + ' crawled images'})
greedy_crawler.crawl(domains=search_url, max_num=1000,
                     min_size=None, max_size=None)

