import urllib.parse
from scrapling import Fetcher
from .BaseAgent import BaseAgent
from models.db_schemas import Project
import asyncio

class CourseAgent(BaseAgent):
    def __init__(self, vectordb_client, generation_client, mongo_client, template_parser, embedding_client):
        super().__init__(vectordb_client, generation_client, mongo_client, template_parser, embedding_client)

    def scrape_online_courses(self, query: str):
        """كشط سريع للروابط المباشرة من منصات التعليم."""
        try:
            fetcher = Fetcher()
            arabic_keywords = ["عربي", "بالعربي", "arabic", "شرح عربي", "مترجم"]
            is_arabic = any(kw in query.lower() for kw in arabic_keywords)
            search_term = f"{query} شرح عربي" if is_arabic else query
            
            sites = "site:coursera.org OR site:udemy.com OR site:datacamp.com OR site:youtube.com/playlist"
            search_query = urllib.parse.quote(f"{search_term} ({sites})")
            url = f"https://www.google.com/search?q={search_query}"
            
            page = fetcher.get(url)
            results = page.css(".tF2Cxc") 
            
            suggestions = []
            for res in results[:5]:
                title_elem = res.css_first("h3")
                link_elem = res.css_first("a")
                if title_elem and link_elem:
                    suggestions.append(f"- {title_elem.text}: {link_elem.attributes.get('href')}")
            
            return "\n".join(suggestions)
        except Exception as e:
            print(f"[DEBUG] Course Scraping Error: {e}")
            return ""

    async def process_stream(self, project, query: str, chat_history: list = [], limit: int = 5):
        """توليد الكورسات بنظام Stream (Stateless)"""
        print(f"\n[DEBUG] === Starting CourseAgent (Stateless) for query: '{query}' ===")

        # 1. البحث والكشط
        arabic_keywords = ["عربي", "بالعربي", "arabic", "شرح عربي"]
        is_arabic = any(kw in query.lower() for kw in arabic_keywords)
        lang_modifier = "in Arabic (شرح عربي)" if is_arabic else ""
        
        enhanced_search_query = f"best online courses or playlists for {query} {lang_modifier} on Coursera, Udemy, DataCamp, YouTube"
        online_results = await self.web_search(enhanced_search_query)
        loop = asyncio.get_event_loop()
        direct_links = await loop.run_in_executor(None, self.scrape_online_courses, query)

        # 2. الدستور (System Prompt)
        system_prompt = (
            "أنت 'الدحيح'، خبير التعلم الذاتي العالمي. "
            "مهمتك هي ترشيح كورسات خارجية من (Coursera, Udemy, DataCamp, YouTube) بأسلوبك الكوميدي التعليمي المشهور. "
            "استخدم جملك الشهيرة مثل: 'يا عزيزي'، 'بص يا غالي'. "
            "إذا لاحظت أن الطالب يريد محتوى 'عربي'، ركز بشدة على ترشيح قنوات يوتيوب عربية أو كورسات مترجمة. "
            "لو الطالب سألك عن الكلية، قوله: 'دي عند دكاترة الكلية يا عزيزي، أنا هنا عشان أفتحلك آفاق العالم الخارجي!'"
        )

        context_aware_query = f"""
إليك نتائج البحث من الإنترنت:
{online_results}

روابط مباشرة مقترحة للكورسات:
{direct_links if direct_links else "لا توجد روابط مباشرة حالياً"}

سؤال الطالب الحالي: {query}
"""

        # 3. معالجة الذاكرة السابقة للـ LLM
        final_chat_history = []
        if chat_history:
            for msg in chat_history:
                final_chat_history.append(
                    self.generation_client.construct_prompt(prompt=msg['content'], role=msg['role'])
                )

        # 4. التوليد (Streaming) - بدون أي تخزين
        async for chunk in self.generation_client.generate_stream(
            prompt=system_prompt,
            user_query=context_aware_query,
            chat_history=final_chat_history,
            temperature=0.75
        ):
            clean_chunk = chunk.replace("</thinking>", "").replace("<thinking>", "")
            if clean_chunk:
                yield clean_chunk