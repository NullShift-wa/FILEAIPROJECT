from icrawler.builtin import BingImageCrawler
import os, time

text = "Ophiophagus hannah"
save_dir = os.path.join(r"D:\AI_PROJECT\Assets\raw", text)
os.makedirs(save_dir, exist_ok=True)

crawler = BingImageCrawler(storage={"root_dir": save_dir}, downloader_threads=2)
crawler.crawl(
    keyword=text,
    max_num=400,
    min_size=None,  
    file_idx_offset=0
)
print("Done.")
