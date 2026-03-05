from scrapling import Fetcher
import urllib.parse
from .BaseAgent import BaseAgent
from models.db_schemas import Project

class JobAgent(BaseAgent):
    
    def scrape_live_jobs(self, query: str) -> str:
        """
        Uses Scrapling to search Wuzzuf for live job postings based on the user's query,
        extracting the job titles and requirements/skills.
        """
        try:
            fetcher = Fetcher()
            
            # Encode the user's query safely for the URL
            safe_query = urllib.parse.quote(query)
            url = f"https://wuzzuf.net/search/jobs/?q={safe_query}"
            
            # Fetch the page using Scrapling
            page = fetcher.get(url)
            
            # Use Scrapling's CSS selectors to find job cards (Wuzzuf's specific HTML classes)
            job_cards = page.css(".css-1gatmva")
            
            if not job_cards:
                return ""
            
            scraped_data = []
            
            # Limit to the top 3 jobs so we don't overwhelm the LLM's context window
            for card in job_cards[:3]:
                # Extract title and skills safely
                title_elem = card.css_first(".css-m604qf")
                skills_elem = card.css_first(".css-y4udm8")
                
                title = title_elem.text if title_elem else "غير محدد"
                skills = skills_elem.text if skills_elem else "غير محدد"
                
                scraped_data.append(f"- المسمى الوظيفي: {title}\n  المهارات/المتطلبات: {skills}")
                
            return "\n".join(scraped_data)
            
        except Exception as e:
            print(f"Scraping failed: {e}")
            return ""

    async def process(self, project, query: str):
        """
        وكيل الوظائف: يدمج البحث في الويب مع كشط النتائج من Wuzzuf.
        """
        # 1. تنظيف السؤال وتجهيزه للبحث
        search_query = urllib.parse.quote(query)
        
        # 2. البحث عن وظائف حية (Scraping)
        # ملاحظة: تم إلغاء الـ self.search هنا لتجنب خطأ الـ Vector Search
        try:
            fetcher = Fetcher()
            # استهداف Wuzzuf مباشرة للبحث عن وظائف في مصر
            url = f"https://wuzzuf.net/search/jobs/?q={search_query}"
            page = fetcher.get(url)
            
            # استخراج العناوين والشركات (Selectors قد تختلف حسب تحديثات الموقع)
            job_titles = page.css(".css-m604qf")
            company_names = page.css(".css-17s97q8")
            
            live_jobs = []
            for i in range(min(len(job_titles), 3)):
                live_jobs.append(f"- وظيفة: {job_titles[i].text} في شركة {company_names[i].text}")
            
            jobs_context = "\n".join(live_jobs) if live_jobs else "لا توجد وظائف حية حالياً."
        except Exception as e:
            self.logger.error(f"Scraping failed: {e}")
            jobs_context = "تعذر جلب وظائف مباشرة من Wuzzuf."

        # 3. البحث العام في الويب (استخدام الـ BaseAgent المصلح)
        web_info = await self.web_search(f"مهارات ومتطلبات {query} في سوق العمل المصري")

        # 4. بناء الرد النهائي بأسلوب احترافي
        system_prompt = (
            "أنت خبير توظيف ومستشار مهني لطلاب الهندسة. "
            "مهمتك هي تحليل متطلبات السوق وتقديم نصائح مهنية بناءً على البيانات المقدمة. "
            "اجعل إجابتك عملية، واذكر المهارات التقنية المطلوبة (Hard Skills) والمهارات الناعمة (Soft Skills)."
        )

        full_prompt = f"""
بناءً على نتائج سوق العمل الحالية:
بيانات الوظائف المتاحة:
{jobs_context}

معلومات إضافية عن المتطلبات:
{web_info}

سؤال الطالب: {query}

الإجابة المهنية:"""

        chat_history = [
            self.generation_client.construct_prompt(prompt=system_prompt, role=self.generation_client.enums.SYSTEM.value)
        ]

        # توليد الرد
        answer = self.generation_client.generate_response(prompt=full_prompt, chat_history=chat_history)
        
        return answer, full_prompt, chat_history