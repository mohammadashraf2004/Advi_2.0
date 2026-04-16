import urllib.parse
import re
import asyncio
import logging
import feedparser          # pip install feedparser --break-system-packages
from scrapling import Fetcher
from .BaseAgent import BaseAgent

logger = logging.getLogger(__name__)


class JobAgent(BaseAgent):
    def __init__(self, vectordb_client, generation_client, mongo_client,
                 template_parser, embedding_client):
        super().__init__(vectordb_client, generation_client, mongo_client,
                         template_parser, embedding_client)
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    # ------------------------------------------------------------------
    # Bug 1 fix — this method was missing entirely
    # ------------------------------------------------------------------
    def _extract_job_title(self, query: str) -> str:
        """استخراج مسمى الوظيفة من السؤال الطبيعي."""
        # English terms are more precise for scraping job boards
        english_terms = re.findall(r'[A-Za-z][A-Za-z\s\+\#\.]{2,}', query)
        if english_terms:
            return max(english_terms, key=len).strip()

        # Strip common Arabic filler words, keep the technical term
        stop_words = {
            'ابحث', 'لي', 'عن', 'وظائف', 'في', 'مجال', 'فرص', 'عمل',
            'هل', 'كيف', 'تدريب', 'فرصة', 'وظيفة', 'وظيفه', 'شغل',
            'internship', 'ايه', 'إيه', 'عايز', 'أوظف', 'أشتغل'
        }
        words = [w for w in query.split() if w not in stop_words]
        return " ".join(words[:4]).strip() or query

    # ------------------------------------------------------------------
    # Bug 4 fix — use feedparser for RSS (css() doesn't work on XML)
    # Bug 3 fix — single Fetcher instance, created once per method call
    # Bug 2 fix — dead HTML scrapers removed, only RSS strategy kept
    # ------------------------------------------------------------------
    def _scrape_multiple_sites(self, job_title: str, location: str) -> str:
        all_jobs = []
        q     = urllib.parse.quote(job_title)
        loc_q = urllib.parse.quote(location)

        # ── Strategy 1: Wuzzuf RSS via feedparser ──────────────────────
        try:
            rss_url = f"https://wuzzuf.net/search/jobs/feed/?q={q}"
            feed    = feedparser.parse(rss_url)

            for entry in feed.entries[:5]:
                title   = entry.get("title", "").strip()
                link    = entry.get("link", "").strip()
                summary = entry.get("summary", "")
                # Company name often appears as the first line of the summary
                company = summary.split("<")[0].strip()[:80] if summary else "غير محدد"

                if title and link:
                    all_jobs.append(
                        f"💼 [Wuzzuf] {title}\n"
                        f"🏢 {company or 'غير محدد'}\n"
                        f"🔗 {link}"
                    )
            logger.info(f"Wuzzuf RSS: {len(all_jobs)} entries found")
        except Exception as e:
            logger.warning(f"Wuzzuf RSS failed: {e}")

        # ── Strategy 2: Bayt RSS via feedparser ────────────────────────
        try:
            country = "egypt" if ("egypt" in location.lower() or "مصر" in location) else "ae"
            rss_url = f"https://www.bayt.com/en/{country}/jobs/rss/?q={q}"
            feed    = feedparser.parse(rss_url)

            bayt_jobs = []
            for entry in feed.entries[:4]:
                title = entry.get("title", "").strip()
                link  = entry.get("link", "").strip()
                if title and link:
                    bayt_jobs.append(
                        f"💼 [Bayt] {title}\n"
                        f"🔗 {link}"
                    )
            all_jobs += bayt_jobs
            logger.info(f"Bayt RSS: {len(bayt_jobs)} entries found")
        except Exception as e:
            logger.warning(f"Bayt RSS failed: {e}")

        # ── Strategy 3: Forasna via Fetcher (HTML, lightweight) ─────────
        if len(all_jobs) < 3:
            try:
                fetcher   = Fetcher()
                forasna_q = urllib.parse.quote(job_title)
                page      = fetcher.get(
                    f"https://forasna.com/jobs?q={forasna_q}",
                    headers=self._headers,
                    timeout=10
                )
                # Forasna renders cards as <div class="job-card"> with an <a> title
                for card in page.css("div.job-card, article.job")[:4]:
                    title_el   = card.css_first("h2 a, h3 a, .job-title a")
                    company_el = card.css_first(".company-name, .employer-name")
                    if not title_el:
                        continue
                    title   = title_el.text.strip()
                    href    = title_el.attributes.get("href", "")
                    link    = f"https://forasna.com{href}" if href.startswith("/") else href
                    company = company_el.text.strip() if company_el else "غير محدد"
                    all_jobs.append(f"💼 [Forasna] {title}\n🏢 {company}\n🔗 {link}")
                logger.info(f"Forasna HTML: found additional jobs, total now {len(all_jobs)}")
            except Exception as e:
                logger.warning(f"Forasna scrape failed: {e}")

        # ── Fallback: direct search links if everything failed ──────────
        if not all_jobs:
            logger.warning("All scrapers failed — returning direct search links")
            return (
                f"لم أجد وظائف مباشرة الآن. إليك روابط البحث المباشر:\n\n"
                f"- **Wuzzuf**: https://wuzzuf.net/search/jobs/?q={q}\n"
                f"- **Bayt**: https://www.bayt.com/en/egypt/jobs/?q={q}\n"
                f"- **LinkedIn**: https://www.linkedin.com/jobs/search?keywords={q}&location={loc_q}\n"
                f"- **Forasna**: https://forasna.com/jobs?q={q}"
            )

        return "\n\n".join(all_jobs)

    # ------------------------------------------------------------------
    # process_stream — unchanged structure, all bugs above now fixed
    # ------------------------------------------------------------------
    async def process_stream(self, project, query, chat_history=None, limit=5):
        if chat_history is None:
            chat_history = []

        logger.info(f"JobAgent query: '{query}'")

        # Location detection
        location_context = "Egypt"
        if "منصورة" in query or "mansoura" in query.lower():
            location_context = "Mansoura, Egypt"
        if "عن بعد" in query or "remote" in query.lower():
            location_context = "Remote"

        # ✅ _extract_job_title now exists
        job_title = self._extract_job_title(query)
        logger.info(f"Extracted: '{job_title}' | Location: '{location_context}'")

        # Run blocking scraper in thread pool
        loop            = asyncio.get_running_loop()
        aggregated_jobs = await loop.run_in_executor(
            None, self._scrape_multiple_sites, job_title, location_context
        )

        web_info = await self.web_search(
            f"أهم المهارات المطلوبة لوظيفة {job_title} في {location_context} 2025"
        )

        system_prompt = """أنت مستشار توظيف خبير. لديك بيانات وظائف حقيقية من منصات متعددة.
قدّم ردك بهذا الترتيب:
1. عرض الوظائف المستخرجة مع شركاتها وروابطها بوضوح.
2. ملخص لأهم 3-5 مهارات مشتركة مطلوبة.
3. نصيحة واحدة عملية للتقديم.
إذا كانت النتائج روابط بحث فقط، وضّح ذلك للطالب واشرح كيف يستخدم الرابط."""

        context_query = f"""[الوظائف المستخرجة]:
{aggregated_jobs}

[تحليل سوق العمل]:
{web_info or 'لا توجد بيانات إضافية.'}

سؤال الطالب: {query}"""

        final_chat_history = [
            self.generation_client.construct_prompt(prompt=system_prompt, role="system")
        ]
        for msg in chat_history:
            final_chat_history.append(
                self.generation_client.construct_prompt(
                    prompt=msg["content"], role=msg["role"]
                )
            )

        async for chunk in self.generation_client.generate_stream(
            prompt=context_query,
            chat_history=final_chat_history,
            temperature=0.3
        ):
            clean = chunk.replace("</thinking>", "").replace("<thinking>", "")
            if clean:
                yield clean