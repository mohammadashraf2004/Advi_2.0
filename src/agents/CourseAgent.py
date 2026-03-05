from scrapling import Fetcher
import urllib.parse
from .BaseAgent import BaseAgent
from models.db_schemas import Project

class CourseAgent(BaseAgent):
    def scrape_online_courses(self, course_name: str):
        """
        استخدام Scrapling للبحث عن كورسات متعلقة بالمادة على المنصات العالمية.
        """
        try:
            fetcher = Fetcher()
            # البحث عن المادة في جوجل مخصص لكورسيرا ويوديمي
            search_query = urllib.parse.quote(f"{course_name} site:coursera.org OR site:udemy.com")
            url = f"https://www.google.com/search?q={search_query}"
            
            page = fetcher.get(url)
            # استخراج العناوين والروابط (تغيير الـ selectors بناءً على هيكل الصفحة)
            results = page.css(".tF2Cxc") 
            
            suggestions = []
            for res in results[:3]: # نأخذ أول 3 نتائج فقط
                title = res.css_first("h3").text if res.css_first("h3") else "كورس مقترح"
                link = res.css_first("a").attributes.get("href")
                suggestions.append(f"- {title}: {link}")
                
            return "\n".join(suggestions)
        except Exception as e:
            self.logger.error(f"Course Scraping Error: {e}")
            return ""
        
    async def process(self, project, query):
        """
        وكيل متخصص في البحث عن الكورسات الخارجية فقط عبر الإنترنت.
        """
        # 1. البحث عن الكورسات عبر الإنترنت مباشرة (تم تخطي الـ VectorDB)
        # نقوم بتحسين الاستعلام ليتم البحث في المنصات التعليمية الكبرى فقط
        enhanced_search_query = f"best online courses for {query} on Coursera Udemy edX YouTube"
        online_results = await self.web_search(enhanced_search_query)

        # 2. إعداد الـ System Prompt (شخصية الدحيح كخبير في التعلم الذاتي)
        system_prompt = (
            "أنت مرشد أكاديمي ذكي بأسلوب 'الدحيح'. وظيفتك هي مساعدة الطلاب في العثور على أفضل الكورسات "
            "على الإنترنت (مثل Coursera و Udemy و YouTube). "
            "ركز تماماً على الروابط والمحتوى الخارجي ولا تتحدث عن لوائح الكلية. "
            "يجب أن تكون إجابتك بالعامية المصرية الراقية، ممتعة، ومليئة بالتشجيع."
        )

        # 3. بناء الـ Full Prompt لإرساله للـ LLM
        full_prompt = f"""
بناءً على نتائج البحث الحالية من الإنترنت:
{online_results}

سؤال الطالب: {query}

إجابة 'الدحيح' المقترحة (مع ذكر الروابط وأهمية كل كورس):"""

        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt, 
                role=self.generation_client.enums.SYSTEM.value
            )
        ]

        # 4. توليد الرد النهائي باستخدام الـ LLM
        answer = self.generation_client.generate_response(
            prompt=full_prompt, 
            chat_history=chat_history
        )
        
        return answer, full_prompt, chat_history