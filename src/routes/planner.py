from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List
from agents.PlannerAgent import GraduationPlanner, PLAN_10_SEMESTER, PLAN_9_SEMESTER, PLAN_8_SEMESTER

planner_router = APIRouter(
    prefix="/api/v1/planner",
    tags=["planner"],
)

DB_PATH = "course_db.json"
planner_engine = GraduationPlanner(DB_PATH)

# ==========================================
# 🌟 الـ Schemas عشان نستقبل الداتا من الواجهة
# ==========================================
class MoveRequest(BaseModel):
    current_plan: Dict[str, List[str]] # الخطة الحالية للطالب
    courses_to_move: Dict[str, List[str]] # المواد اللي اختارها { "1": ["BAS 011"] }
    to_semester: str # الفصل اللي هينقل ليه

@planner_router.get("/plan/{plan_type}")
async def get_graduation_plan(plan_type: str):
    if plan_type == "10":
        plan_data = planner_engine.get_full_plan_details(PLAN_10_SEMESTER)
    elif plan_type == "9":
        plan_data = planner_engine.get_full_plan_details(PLAN_9_SEMESTER)
    elif plan_type == "8":
        plan_data = planner_engine.get_full_plan_details(PLAN_8_SEMESTER)
    else:
        return JSONResponse(status_code=400, content={"error": "Invalid plan type."})
        
    return JSONResponse(content=plan_data)

@planner_router.post("/move")
async def move_courses(request: MoveRequest):
    """
    الـ Endpoint الجديدة لتطبيق لوجيك النقل بتاع Gradio بس كـ API
    """
    # 1. تحميل الخطة الحالية للطالب
    planner_engine.load_plan(request.current_plan)
    
    # 2. تنفيذ النقل
    move_message = planner_engine.move_courses_list(request.courses_to_move, request.to_semester)
    
    # 3. إعادة الفحص (Validation) واستخراج الخطة المحدثة
    validation = planner_engine.validate_plan()
    
    detailed_plan = {}
    for sem, courses in planner_engine.plan.items():
        sem_details = []
        total_credits = 0
        for code in courses:
            info = planner_engine.get_course_info(code)
            if info:
                sem_details.append({"code": code, "name": info["name"], "credits": info["credits"]})
                total_credits += info["credits"]
        detailed_plan[sem] = {"courses": sem_details, "total_credits": total_credits}

    return JSONResponse(content={
        "message": move_message,
        "validation": validation,
        "plan": detailed_plan,
        "raw_plan": planner_engine.plan # نرجعها عشان الواجهة تحتفظ بيها للعملية الجاية
    })