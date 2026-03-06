from flask import Blueprint, request, jsonify
from services.fund_intel.clients import NewsClient, LLMClient
from shared.messaging import get_mq, NEWS_CRAWLED
import threading
import redis

news_classification_bp = Blueprint("news_classification", __name__)
news_client = NewsClient()
llm_client = LLMClient()


def _start_news_listener():
    r = redis.from_url("redis://redis:6379", decode_responses=True)

    try:
        r.xgroup_create(NEWS_CRAWLED, "classification_group", id="0", mkstream=True)
    except redis.ResponseError:
        pass

    while True:
        try:
            messages = r.xreadgroup(
                "classification_group",
                "classification_consumer",
                {NEWS_CRAWLED: ">"},
                count=1,
                block=5000,
            )

            if messages:
                for stream, msgs in messages:
                    for msg_id, msg in msgs:
                        try:
                            news_list = news_client.get_news(days=1)
                            for news in news_list[:10]:
                                title = news.get("title", "")
                                content = news.get("content", "")
                                if title:
                                    llm_client.classify_industry(
                                        f"{title} {content[:200]}"
                                    )
                        except:
                            pass
                        r.xack(NEWS_CRAWLED, "classification_group", msg_id)
        except Exception as e:
            print(f"Listener error: {e}")
            import time

            time.sleep(5)


listener_thread = threading.Thread(target=_start_news_listener, daemon=True)
listener_thread.start()


@news_classification_bp.route("/classify", methods=["POST"])
def classify_news():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"success": False, "message": "Text required"}), 400

    industry = llm_client.classify_industry(text)

    return jsonify({"success": True, "data": {"text": text, "industry": industry}})


@news_classification_bp.route("/classify-today", methods=["POST"])
def classify_today_news():
    news_list = news_client.get_news(days=1)

    classified = []
    for news in news_list[:50]:
        title = news.get("title", "")
        content = news.get("content", "")
        if title:
            industry = llm_client.classify_industry(f"{title} {content[:200]}")
            classified.append({"id": news.get("id"), "industry": industry})

    return jsonify(
        {
            "success": True,
            "data": {
                "total": len(news_list),
                "classified": len(classified),
                "results": classified,
            },
        }
    )


@news_classification_bp.route("/industries", methods=["GET"])
def get_industries():
    return jsonify({"success": True, "data": ["宏观", "行业", "全球", "政策", "公司"]})


@news_classification_bp.route("/industry/<industry>", methods=["GET"])
def get_news_by_industry(industry):
    days = int(request.args.get("days", 7))
    news_list = news_client.get_news_by_industry(industry, days=days)

    return jsonify({"success": True, "data": news_list})


@news_classification_bp.route("/stats", methods=["GET"])
def get_industry_stats():
    industries = ["宏观", "行业", "全球", "政策", "公司"]
    stats = {}

    for ind in industries:
        news = news_client.get_news_by_industry(ind, days=7)
        stats[ind] = len(news)

    return jsonify({"success": True, "data": stats})


@news_classification_bp.route("/today", methods=["GET"])
def get_today_classified():
    news_list = news_client.get_news(days=1)

    classified = []
    for news in news_list[:20]:
        title = news.get("title", "")
        content = news.get("content", "")
        if title:
            industry = llm_client.classify_industry(f"{title} {content[:200]}")
            classified.append(
                {"id": news.get("id"), "title": title, "industry": industry}
            )

    return jsonify({"success": True, "data": classified})
