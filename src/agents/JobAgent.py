import os
import json
import asyncio
import logging
import urllib.parse
import feedparser
import re
from .BaseAgent import BaseAgent

logger = logging.getLogger(__name__)


class JobAgent(BaseAgent):
    """
    JobAgent — scrapes Wuzzuf, Bayt, Indeed, LinkedIn for real listings.
    Detects internship vs full-time, AI/ML domain, location.
    Generates personalised CV advice per student context.
    """

    def __init__(self, vectordb_client, generation_client, mongo_client,
                 template_parser, embedding_client):
        super().__init__(vectordb_client, generation_client, mongo_client,
                         template_parser, embedding_client)
                         
        # Tell the Orchestrator NOT to attempt Qdrant vector caching on this agent
        self.is_vector_agent = False

        self._ai_keywords = {
            "machine learning","deep learning","data science","ai engineer",
            "nlp","computer vision","ذكاء اصطناعي","تعلم آلة","بيانات",
            "data analyst","data engineer","mlops","llm","python developer",
            "pytorch","tensorflow","neural"
        }
        self._internship_keywords = {
            "internship","تدريب","تدريب صيفي","intern","summer training",
            "تدريب عملي","فترة تدريب","تدريبات","تدريب ميداني"
        }

    def _analyse_query(self, query: str) -> dict:
        q = query.lower()
        is_internship = any(kw in q for kw in self._internship_keywords)
        is_ai         = any(kw in q for kw in self._ai_keywords)

        location = "Egypt"
        if any(w in q for w in ["منصورة","mansoura"]):       location = "Mansoura, Egypt"
        elif any(w in q for w in ["قاهرة","cairo"]):          location = "Cairo, Egypt"
        elif any(w in q for w in ["إسكندرية","alexandria"]):  location = "Alexandria, Egypt"
        elif any(w in q for w in ["عن بعد","remote","اونلاين"]): location = "Remote"

        english = re.findall(r'[A-Za-z][A-Za-z\s\+\#\.]{2,}', query)
        if english:
            job_title = max(english, key=len).strip()
        else:
            stop = {'ابحث','لي','عن','وظائف','في','مجال','فرص','عمل','هل',
                    'كيف','تدريب','internship','فرصة','وظيفة','شغل','ايه','ما'}
            job_title = " ".join(
                [w for w in query.split() if w not in stop][:4]
            ).strip() or query

        return {"job_title":job_title,"location":location,
                "is_internship":is_internship,"is_ai":is_ai}

    def _scrape_wuzzuf(self, job_title, is_internship):
        jobs = []
        try:
            search = f"{job_title} internship" if is_internship else job_title
            q = urllib.parse.quote(search)
            feed = feedparser.parse(f"https://wuzzuf.net/search/jobs/feed/?q={q}&a=hpb")
            for e in feed.entries[:6]:
                title = e.get("title","").strip()
                link  = e.get("link","").strip()
                if not title or not link: continue
                summary   = e.get("summary","")
                company_m = re.search(r'<b>([^<]+)</b>', summary)
                company   = company_m.group(1).strip() if company_m else "غير محدد"
                salary_m  = re.search(r'(\d[\d,]+)\s*(?:EGP|جنيه|LE)', summary, re.I)
                salary    = f" — 💰 {salary_m.group(0)}" if salary_m else ""
                jobs.append({"source":"Wuzzuf","title":title,"company":company,
                             "link":link,"salary":salary})
            logger.info(f"Wuzzuf: {len(jobs)}")
        except Exception as e:
            logger.warning(f"Wuzzuf: {e}")
        return jobs

    def _scrape_bayt(self, job_title, is_internship):
        jobs = []
        try:
            search = f"{job_title} internship" if is_internship else job_title
            q = urllib.parse.quote(search)
            feed = feedparser.parse(
                f"https://www.bayt.com/en/egypt/jobs/?q={q}&rss=1")
            for e in feed.entries[:5]:
                raw  = e.get("title","").strip()
                link = e.get("link","").strip()
                if not raw or not link: continue
                if " at " in raw:
                    title, company = raw.split(" at ",1)
                    title, company = title.strip(), company.strip()
                else:
                    title, company = raw, "غير محدد"
                jobs.append({"source":"Bayt","title":title,"company":company,
                             "link":link,"salary":""})
            logger.info(f"Bayt: {len(jobs)}")
        except Exception as e:
            logger.warning(f"Bayt: {e}")
        return jobs

    def _scrape_indeed(self, job_title, is_internship):
        jobs = []
        try:
            search = f"{job_title} internship" if is_internship else job_title
            q   = urllib.parse.quote(search)
            loc = urllib.parse.quote("Egypt")
            feed = feedparser.parse(
                f"https://www.indeed.com/rss?q={q}&l={loc}&sort=date&limit=10")
            for e in feed.entries[:6]:
                title = e.get("title","").strip()
                link  = e.get("link","").strip()
                if not title or not link: continue
                if " - " in title:
                    parts = title.rsplit(" - ",1)
                    title, company = parts[0].strip(), parts[1].strip()
                else:
                    company = e.get("author","غير محدد")
                summary  = e.get("summary","")
                salary_m = re.search(
                    r'(\$[\d,]+|\d[\d,]+\s*(?:EGP|LE|جنيه))', summary, re.I)
                salary = f" — 💰 {salary_m.group(0)}" if salary_m else ""
                jobs.append({"source":"Indeed","title":title,"company":company,
                             "link":link,"salary":salary})
            logger.info(f"Indeed: {len(jobs)}")
        except Exception as e:
            logger.warning(f"Indeed: {e}")
        return jobs

    def _linkedin_link(self, job_title, location, is_internship):
        q   = urllib.parse.quote(job_title)
        loc = urllib.parse.quote(location)
        exp = "&f_E=1" if is_internship else ""
        url = (f"https://www.linkedin.com/jobs/search/?keywords={q}"
               f"&location={loc}&f_TPR=r604800{exp}")
        return [{"source":"LinkedIn",
                 "title":f"ابحث عن '{job_title}' (آخر 7 أيام)",
                 "company":"بحث مباشر","link":url,"salary":""}]

    def _ai_boards(self, job_title):
        q = urllib.parse.quote(job_title)
        return [
            {"source":"AI Jobs","title":f"وظائف AI/ML — {job_title}",
             "company":"ai-jobs.net","link":f"https://ai-jobs.net/?search={q}","salary":""},
            {"source":"Kaggle","title":f"وظائف {job_title} على Kaggle",
             "company":"Kaggle Jobs","link":f"https://www.kaggle.com/jobs?q={q}","salary":""},
        ]

    def _format_jobs(self, jobs):
        if not jobs: return "لم يتم العثور على نتائج."
        return "\n\n".join(
            f"💼 **[{j['source']}] {j['title']}**{j.get('salary','')}\n"
            f"🏢 {j['company']}\n🔗 {j['link']}"
            for j in jobs
        )

    def _scrape_all(self, info):
        t, loc = info["job_title"], info["location"]
        intern, is_ai = info["is_internship"], info["is_ai"]

        all_j = (self._scrape_wuzzuf(t, intern)
               + self._scrape_bayt(t, intern)
               + self._scrape_indeed(t, intern)
               + self._linkedin_link(t, loc, intern)
               + (self._ai_boards(t) if is_ai else []))

        if not any(j["source"] in ("Wuzzuf","Bayt","Indeed")
                   for j in all_j):
            q = urllib.parse.quote(t)
            return (
                "⚠️ لم تُرجع المنصات نتائج مباشرة. روابط البحث:\n\n"
                f"- **Wuzzuf**: https://wuzzuf.net/search/jobs/?q={q}\n"
                f"- **Indeed**: https://www.indeed.com/jobs?q={q}&l=Egypt\n"
                f"- **Bayt**: https://www.bayt.com/en/egypt/jobs/?q={q}\n"
                f"- **LinkedIn**: https://linkedin.com/jobs/search?keywords={q}\n"
                + (f"- **AI Jobs**: https://ai-jobs.net/?search={q}\n" if is_ai else "")
            )
        return self._format_jobs(all_j)

    def _cv_advice(self, info):
        if info["is_internship"]:
            return (
                "**نصائح التقديم للتدريب الصيفي:**\n"
                "- ✅ المشاريع الجامعية + GitHub أهم من GPA\n"
                "- ✅ اكتب المقررات ذات الصلة + أي مشروع تقني ولو صغير\n"
                "- ✅ وقت التقديم المثالي: يناير–مارس لتدريبات الصيف\n"
                "- ✅ تواصل مباشرة مع HR على LinkedIn برسالة قصيرة\n"
                "- ⚠️ اكتب الـ CV بالإنجليزية حتى للشركات المصرية"
            )
        if info["is_ai"]:
            return (
                "**نصائح التقديم لوظائف AI/ML:**\n"
                "- ✅ GitHub + Kaggle profile أهم من أي شهادة\n"
                "- ✅ المهارات الأساسية: Python · PyTorch/TF · SQL · Git · Docker\n"
                "- ✅ اكتب نتائج قابلة للقياس: 'رفعت accuracy من 78% لـ91%'\n"
                "- ✅ LinkedIn headline: 'AI Engineer | Deep Learning | Python'\n"
                "- ⚠️ أضف رابط GitHub في أول سطر في الـ CV\n"
                "- ⚠️ شركات مصر تطلب Deployment (FastAPI, Docker) مش بس modeling"
            )
        return (
            "**نصائح التقديم العامة:**\n"
            "- ✅ خصّص الـ CV لكل وظيفة — استخدم كلمات وصف الوظيفة نفسها\n"
            "- ✅ Cover letter قصير (3 أسطر) يرفع فرصة الرد 40%\n"
            "- ✅ التقديم المباشر على موقع الشركة أفضل من المنصات\n"
            "- ✅ Follow up على LinkedIn بعد أسبوع إن لم يردوا\n"
            "- ⚠️ لا تكتب 'خبرة: 0' — اكتب 'حديث التخرج' بدلاً منها"
        )

    async def process_stream(self, project, query: str,
                             chat_history: list = None, limit: int = 5, skip_evaluation: bool = False, **kwargs):
        if chat_history is None:
            chat_history = []

        # Check if we are in voice mode (from Orchestrator)
        voice_mode = kwargs.get('voice_mode', False)

        logger.info(f"JobAgent: '{query}'")
        info   = self._analyse_query(query)
        logger.info(f"→ {info}")

        loop   = asyncio.get_running_loop()
        jobs   = await loop.run_in_executor(None, self._scrape_all, info)
        advice = self._cv_advice(info)

        intern_note = "الطالب يبحث عن تدريب صيفي.\n" if info["is_internship"] else ""
        ai_note     = "الطالب في مجال AI/ML — أبرز المهارات التقنية.\n" if info["is_ai"] else ""

        # Safety instruction to prevent Gemini from "speaking" a long URL in Voice Mode
        voice_rule = ""
        if voice_mode:
            voice_rule = (
                "🚨 **تحذير للوضع الصوتي:** لا تقم بكتابة أو نطق أي روابط (URLs) نهائياً. "
                "اكتفِ بذكر أسماء المنصات (مثل Wuzzuf و LinkedIn) وقل للطالب أن يبحث فيها."
            )

        system_prompt = (
            "أنت مستشار توظيف متخصص في سوق العمل المصري لطلاب هندسة الذكاء الاصطناعي.\n"
            f"{intern_note}{ai_note}\n"
            "قدّم ردك بهذا الترتيب:\n"
            "1. **الوظائف المتاحة** — كل وظيفة: عنوانها، شركتها، رابطها (فقط في حالة وضع الشات).\n"
            "2. **المهارات المطلوبة** — أبرز 3-5 مهارات من الوظائف.\n"
            "3. **نصائح التقديم** — استخدم النصائح من السياق كما هي.\n"
            "4. **الخطوة التالية** — جملة واحدة عملية تحفيزية.\n\n"
            "قواعد: لا تخترع وظائف. إذا كانت نتائج فقط روابط، اشرح للطالب كيف يستخدمها.\n"
            f"{voice_rule}"
        )

        context = (
            f"[الوظائف]:\n{jobs}\n\n"
            f"[نصائح التقديم]:\n{advice}\n\n"
            f"[سؤال الطالب]:\n{query}"
        )

        history = [
            self.generation_client.construct_prompt(prompt=system_prompt, role="system")
        ]
        for msg in chat_history:
            history.append(
                self.generation_client.construct_prompt(
                    prompt=msg["content"], role=msg["role"]
                )
            )

        # ✅ FIX: Explicitly set max_output_tokens=4096 to prevent URL cutoffs
        async for chunk in self.generation_client.generate_stream(
            prompt=context, 
            chat_history=history, 
            temperature=0.2,
            max_output_tokens=4096 
        ):
            clean = chunk.replace("<thinking>","").replace("</thinking>","")
            if clean:
                yield clean