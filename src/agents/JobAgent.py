import re
import asyncio
import logging
import urllib.parse
import feedparser
from .BaseAgent import BaseAgent

logger = logging.getLogger(__name__)


class JobAgent(BaseAgent):
    def __init__(self, vectordb_client, generation_client, mongo_client,
                 template_parser, embedding_client):
        super().__init__(vectordb_client, generation_client, mongo_client,
                         template_parser, embedding_client)

    # ── Extract clean job title from natural language query ───────────────
    def _extract_job_title(self, query: str) -> str:
        # English terms work better for job scraping (LinkedIn, Wuzzuf, Bayt)
        english = re.findall(r'[A-Za-z][A-Za-z\s\+\#\.]{2,}', query)
        if english:
            return max(english, key=len).strip()

        # Strip Arabic filler words and return core noun
        stop = {
            'ابحث','لي','عن','وظائف','في','مجال','فرص','عمل','هل','كيف',
            'تدريب','internship','فرصة','وظيفة','وظيفه','شغل','ايه','ما'
        }
        words = [w for w in query.split() if w not in stop]
        return " ".join(words[:4]).strip() or query

    # ── Wuzzuf RSS ────────────────────────────────────────────────────────
    def _scrape_wuzzuf_rss(self, job_title: str) -> list:
        jobs = []
        try:
            q   = urllib.parse.quote(job_title)
            url = f"https://wuzzuf.net/search/jobs/feed/?q={q}&a=hpb"
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                title   = entry.get("title", "").strip()
                link    = entry.get("link", "").strip()
                summary = entry.get("summary", "")
                # Extract company from summary
                company_m = re.search(r'<b>([^<]+)</b>', summary)
                company   = company_m.group(1).strip() if company_m else "غير محدد"
                if title and link:
                    jobs.append(f"💼 **[Wuzzuf] {title}**\n🏢 {company}\n🔗 {link}")
            logger.info(f"Wuzzuf RSS: {len(jobs)} jobs")
        except Exception as e:
            logger.warning(f"Wuzzuf RSS failed: {e}")
        return jobs

    # ── Bayt RSS ──────────────────────────────────────────────────────────
    def _scrape_bayt_rss(self, job_title: str) -> list:
        jobs = []
        try:
            q   = urllib.parse.quote(job_title)
            url = f"https://www.bayt.com/en/egypt/jobs/?q={q}&rss=1"
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                title   = entry.get("title", "").strip()
                link    = entry.get("link",  "").strip()
                # Company is usually in the title: "Job Title at Company"
                if " at " in title:
                    parts   = title.split(" at ", 1)
                    title   = parts[0].strip()
                    company = parts[1].strip()
                else:
                    company = "غير محدد"
                if title and link:
                    jobs.append(f"💼 **[Bayt] {title}**\n🏢 {company}\n🔗 {link}")
            logger.info(f"Bayt RSS: {len(jobs)} jobs")
        except Exception as e:
            logger.warning(f"Bayt RSS failed: {e}")
        return jobs

    # ── LinkedIn public search (no auth) ──────────────────────────────────
    def _scrape_linkedin(self, job_title: str, location: str) -> list:
        """
        LinkedIn blocks scraping but their public jobs search page is accessible.
        We use feedparser on their job alert RSS (no login needed for Egypt).
        Falls back to a direct search link if feed is empty.
        """
        jobs = []
        try:
            q    = urllib.parse.quote(job_title)
            loc  = urllib.parse.quote(location)
            # LinkedIn public RSS feed for job search (works without login)
            url  = (
                f"https://www.linkedin.com/jobs/search/?keywords={q}"
                f"&location={loc}&f_TPR=r604800"  # last 7 days
            )
            # LinkedIn doesn't expose RSS publicly — provide direct search link
            # Try feedparser on their undocumented feed endpoint
            rss_url = f"https://www.linkedin.com/jobs/search.rss?keywords={q}&location={loc}&trk=api"
            feed    = feedparser.parse(rss_url)

            if feed.entries:
                for entry in feed.entries[:5]:
                    title   = entry.get("title", "").strip()
                    link    = entry.get("link",  "").strip()
                    company = entry.get("author", "غير محدد")
                    if title and link:
                        jobs.append(f"💼 **[LinkedIn] {title}**\n🏢 {company}\n🔗 {link}")
                logger.info(f"LinkedIn RSS: {len(jobs)} jobs")
            else:
                # LinkedIn RSS unavailable — provide direct search link
                jobs.append(
                    f"💼 **[LinkedIn] ابحث عن '{job_title}' في {location}**\n"
                    f"🔗 {url}"
                )
                logger.info("LinkedIn: provided direct search link")
        except Exception as e:
            logger.warning(f"LinkedIn scrape failed: {e}")
            q   = urllib.parse.quote(job_title)
            loc = urllib.parse.quote(location)
            jobs.append(
                f"💼 **[LinkedIn] ابحث مباشرة**\n"
                f"🔗 https://www.linkedin.com/jobs/search/?keywords={q}&location={loc}"
            )
        return jobs

    # ── Master scraper ────────────────────────────────────────────────────
    def _scrape_all(self, job_title: str, location: str) -> str:
        logger.info(f"Scraping: '{job_title}' in '{location}'")

        wuzzuf_jobs  = self._scrape_wuzzuf_rss(job_title)
        bayt_jobs    = self._scrape_bayt_rss(job_title)
        linkedin_jobs = self._scrape_linkedin(job_title, location)

        all_jobs = wuzzuf_jobs + bayt_jobs + linkedin_jobs

        if not any(wuzzuf_jobs + bayt_jobs):
            # All RSS failed — provide fallback direct links
            q   = urllib.parse.quote(job_title)
            loc = urllib.parse.quote(location)
            return (
                f"⚠️ لم تُرجع الكاشطات نتائج مباشرة الآن. إليك روابط البحث:\n\n"
                f"- **Wuzzuf**: https://wuzzuf.net/search/jobs/?q={q}\n"
                f"- **Bayt**: https://www.bayt.com/en/egypt/jobs/?q={q}\n"
                f"- **LinkedIn**: https://www.linkedin.com/jobs/search?keywords={q}&location={loc}\n"
                f"- **Indeed**: https://eg.indeed.com/jobs?q={urllib.parse.quote(job_title)}"
            )

        return "\n\n".join(all_jobs)

    # ── process_stream ────────────────────────────────────────────────────
    async def process_stream(self, project, query: str,
                             chat_history: list = None, limit: int = 5):
        """
        NOTE: JobAgent does NOT accept skip_evaluation.
        That parameter only exists on VectorDBAgent.
        """
        if chat_history is None:
            chat_history = []

        logger.info(f"JobAgent query: '{query}'")

        # Detect location from query
        location = "Egypt"
        if "منصورة" in query or "mansoura" in query.lower():
            location = "Mansoura, Egypt"
        elif "قاهرة" in query or "cairo" in query.lower():
            location = "Cairo, Egypt"
        elif "عن بعد" in query or "remote" in query.lower():
            location = "Egypt (Remote)"

        job_title = self._extract_job_title(query)
        logger.info(f"Extracted: '{job_title}' | location: '{location}'")

        # Run scraping in thread pool (blocking IO, don't freeze event loop)
        loop            = asyncio.get_running_loop()
        aggregated_jobs = await loop.run_in_executor(
            None, self._scrape_all, job_title, location
        )

        system_prompt = """أنت مستشار توظيف خبير. لديك بيانات وظائف حقيقية من منصات متعددة.
قدّم ردك بهذا الترتيب:
1. عرض الوظائف المستخرجة مع شركاتها وروابطها بوضوح.
2. ملخص لأهم 3-5 مهارات مشتركة مطلوبة.
3. نصيحة واحدة عملية للتقديم.
إذا كانت الوظائف روابط بحث فقط (بسبب فشل الكاشطات)، وضّح ذلك للطالب واشرح كيف يستخدم الرابط."""

        context_query = f"""[الوظائف المستخرجة]:
{aggregated_jobs}

سؤال الطالب: {query}"""

        final_history = [
            self.generation_client.construct_prompt(prompt=system_prompt, role="system")
        ]
        for msg in chat_history:
            final_history.append(
                self.generation_client.construct_prompt(
                    prompt=msg["content"], role=msg["role"]
                )
            )

        async for chunk in self.generation_client.generate_stream(
            prompt=context_query,
            chat_history=final_history,
            temperature=0.3
        ):
            clean = chunk.replace("<thinking>", "").replace("</thinking>", "")
            if clean:
                yield clean