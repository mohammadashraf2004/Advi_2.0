import os
import json

# ==============================================================
# 🎓 GRADUATION PLANNER ENGINE
# ==============================================================

def setup_course_db(db_path):
    if os.path.exists(db_path):
        return

    course_data = {
        "total_required_hours": 160,
        "courses": {
            "BAS 011": {"name": "رياضيات (1)", "credits": 3, "prerequisites": []},
            "BAS 020": {"name": "ميكانيكا (1)", "credits": 3, "prerequisites": []},
            "BAS 031": {"name": "فيزياء (1)", "credits": 3, "prerequisites": []},
            "BAS 041": {"name": "كيمياء هندسية", "credits": 3, "prerequisites": []},
            "PDE 052": {"name": "رسم هندسي", "credits": 3, "prerequisites": []},
            "UNR 061": {"name": "لغة إنجليزية (1)", "credits": 2, "prerequisites": []},
            "BAS 012": {"name": "رياضيات (2)", "credits": 3, "prerequisites": ["BAS 011"]},
            "BAS 021": {"name": "ميكانيكا (2)", "credits": 3, "prerequisites": ["BAS 020"]},
            "BAS 032": {"name": "فيزياء (2)", "credits": 3, "prerequisites": ["BAS 031"]},
            "CSE 042": {"name": "مقدمة لنظم الحاسب", "credits": 3, "prerequisites": []},
            "PDE 051": {"name": "هندسة الإنتاج", "credits": 3, "prerequisites": []},
            "UNR 021": {"name": "تاريخ الهندسة والتكنولوجيا", "credits": 2, "prerequisites": []},
            "BAS 115": {"name": "الجبر الخطي", "credits": 3, "prerequisites": ["BAS 012"]},
            "CSE 151": {"name": "مقدمة للذكاء الاصطناعي", "credits": 3, "prerequisites": []},
            "CSE 141": {"name": "تصميم رقمي", "credits": 3, "prerequisites": ["CSE 042"]},
            "UNR 181": {"name": "القانون وحقوق الإنسان", "credits": 2, "prerequisites": []},
            "ECE 121": {"name": "دوائر كهربية", "credits": 3, "prerequisites": ["BAS 032"]},
            "ENG 111": {"name": "كتابة تقارير فنية", "credits": 2, "prerequisites": ["UNR 061"]},
            "BAS 116": {"name": "طرق رياضية للمهندسين", "credits": 3, "prerequisites": ["BAS 115"]},
            "ECE 122": {"name": "إلكترونيات", "credits": 3, "prerequisites": ["ECE 121"]},
            "CSE 111": {"name": "برمجة (1)", "credits": 3, "prerequisites": ["CSE 141"]},
            "CSE 112": {"name": "خوارزميات وهياكل بيانات", "credits": 3, "prerequisites": ["CSE 042"]},
            "ELE 151": {"name": "قوى وآلات كهربية", "credits": 3, "prerequisites": ["ECE 121"]},
            "UNR 121": {"name": "مهارات البحث والتحليل", "credits": 2, "prerequisites": []},
            "ARI 171": {"name": "تدريب عملي", "credits": 0, "prerequisites": [], "constraints": ["Pass/Fail"]},
            "BAS 216": {"name": "الإحصاء وتحليل البيانات", "credits": 2, "prerequisites": ["BAS 115"]},
            "ECE 234": {"name": "إشارات ونظم", "credits": 3, "prerequisites": ["BAS 116"]},
            "UNR 241": {"name": "مهارات الاتصال والعرض", "credits": 2, "prerequisites": []},
            "ECE 223": {"name": "قياسات وأجهزة قياس", "credits": 3, "prerequisites": ["ECE 122"]},
            "CSE 251": {"name": "تعلم الآلة", "credits": 3, "prerequisites": ["CSE 151"]},
            "CSE 221": {"name": "تحكم آلي", "credits": 3, "prerequisites": ["BAS 116"]},
            "BAS 217": {"name": "الرياضيات المتقطعة", "credits": 3, "prerequisites": ["BAS 216"]},
            "ECE 224": {"name": "مستشعرات ومؤثرات", "credits": 3, "prerequisites": ["ECE 223"]},
            "BAS 218": {"name": "رياضيات هندسية متقدمة", "credits": 3, "prerequisites": ["BAS 216"]},
            "UNR 261": {"name": "آداب وأخلاقيات المهنة", "credits": 2, "prerequisites": []},
            "CSE 212": {"name": "أنظمة قواعد البيانات", "credits": 3, "prerequisites": ["CSE 112"]},
            "ECE 235": {"name": "معالجة وتحليل الإشارات", "credits": 3, "prerequisites": ["ECE 234"]},
            "ARI 271": {"name": "تدريب (1)", "credits": 0, "prerequisites": ["ARI 171"], "constraints": ["Pass/Fail"]},
            "Elective 1": {"name": "مقرر اختياري (1)", "credits": 3, "prerequisites": []},
            "ECE 332": {"name": "شبكات عصبونية", "credits": 3, "prerequisites": ["BAS 218"]},
            "CSE 311": {"name": "برمجة (2)", "credits": 3, "prerequisites": ["CSE 111", "CSE 212"]},
            "CSE 313": {"name": "إدارة البيانات", "credits": 3, "prerequisites": ["CSE 212"]},
            "CSE 317": {"name": "معمار الحاسب", "credits": 3, "prerequisites": ["CSE 141"]},
            "ECE 333": {"name": "معالجة صور رقمية", "credits": 3, "prerequisites": ["ECE 235"]},
            "CSE 351": {"name": "التعلم العميق", "credits": 3, "prerequisites": ["ECE 332"]},
            "CSE 315": {"name": "الأنظمة المتضمنة", "credits": 3, "prerequisites": ["CSE 317"]},
            "Elective 2": {"name": "مقرر اختياري (2)", "credits": 3, "prerequisites": []},
            "ECE 321": {"name": "شبكات الاتصالات", "credits": 3, "prerequisites": ["ECE 234"]},
            "ENG 312": {"name": "إدارة المشروعات", "credits": 2, "prerequisites": []},
            "ARI 381": {"name": "مشروع (1)", "credits": 3, "prerequisites": [], "constraints": ["Cannot be taken in Summer", "Level 300"]},
            "ARI 371": {"name": "تدريب (2)", "credits": 0, "prerequisites": ["ARI 271"], "constraints": ["Pass/Fail"]},
            "Elective 3": {"name": "مقرر اختياري (3)", "credits": 3, "prerequisites": []},
            "Elective 4": {"name": "مقرر اختياري (4)", "credits": 3, "prerequisites": []},
            "CSE 423": {"name": "روبوتكس", "credits": 3, "prerequisites": ["CSE 221"]},
            "UNR 471": {"name": "التسويق", "credits": 2, "prerequisites": []},
            "ARI 481": {"name": "مشروع (2)", "credits": 3, "prerequisites": ["ARI 381"], "constraints": ["Cannot be taken in Summer", "Level 400"]},
            "CSE 451": {"name": "علم البيانات الكبيرة", "credits": 3, "prerequisites": ["CSE 313"]},
            "CSE 452": {"name": "تطبيقات في الذكاء الاصطناعي", "credits": 3, "prerequisites": ["CSE 351"]},
            "Elective 5": {"name": "مقرر اختياري (5)", "credits": 3, "prerequisites": []},
            "ARI 482": {"name": "مشروع (3)", "credits": 3, "prerequisites": ["ARI 481"], "constraints": ["Cannot be taken in Summer"]}
        }
    }
    total_credits = sum(course['credits'] for course in course_data['courses'].values())
    course_data['total_required_hours'] = total_credits

    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(course_data, f, indent=4, ensure_ascii=False)

PLAN_10_SEMESTER = {
    "1": ["BAS 011", "BAS 020", "BAS 031", "BAS 041", "PDE 052", "UNR 061"],
    "2": ["BAS 012", "BAS 021", "BAS 032", "CSE 042", "PDE 051", "UNR 021"],
    "3": ["BAS 115", "CSE 151", "CSE 141", "UNR 181", "ECE 121", "ENG 111"],
    "4": ["BAS 116", "ECE 122", "CSE 111", "CSE 112", "ELE 151", "UNR 121", "ARI 171"],
    "S2": [],
    "5": ["BAS 216", "ECE 234", "UNR 241", "ECE 223", "CSE 251", "CSE 221"],
    "6": ["BAS 217", "ECE 224", "BAS 218", "UNR 261", "CSE 212", "ECE 235", "ARI 271"],
    "S3": [],
    "7": ["Elective 1", "ECE 332", "CSE 311", "CSE 313", "CSE 317", "ECE 333"],
    "8": ["CSE 351", "CSE 315", "Elective 2", "ECE 321", "ENG 312", "ARI 381", "ARI 371"],
    "S4": [],
    "9": ["Elective 3", "Elective 4", "CSE 423", "UNR 471", "ARI 481"],
    "10": ["CSE 451", "CSE 452", "Elective 5", "ARI 482"]
}

PLAN_9_SEMESTER = {
    "1": PLAN_10_SEMESTER["1"], 
    "2": PLAN_10_SEMESTER["2"], 
    "3": PLAN_10_SEMESTER["3"], 
    "4": PLAN_10_SEMESTER["4"], 
    "S2": ["UNR 241", "UNR 261"], 
    "5": ["BAS 216", "ECE 234", "ECE 223", "CSE 251", "CSE 221"], 
    "6": ["BAS 217", "ECE 224", "BAS 218", "CSE 212", "ECE 235", "ARI 271","ARI 381"], 
    "S3": ["UNR 121", "ENG 312"], 
    "7": PLAN_10_SEMESTER["7"], 
    "8": ["CSE 351", "CSE 315", "Elective 2", "ECE 321", "ARI 371", "UNR 471","ARI 481"], 
    "S4": ["Elective 3"], 
    "9": ["CSE 423","Elective 4", "CSE 451", "CSE 452", "Elective 5", "ARI 482"], 
    "10": [] 
}

PLAN_8_SEMESTER = {
    "1": PLAN_10_SEMESTER["1"], 
    "2": ["BAS 012", "BAS 021", "BAS 032", "CSE 042", "PDE 051", "UNR 021","UNR 181"], 
    "3":["BAS 115", "CSE 151", "CSE 141", "ECE 121", "ENG 111","UNR 241","UNR 261"], 
    "4": ["BAS 116","ECE 122", "CSE 111", "CSE 112", "ELE 151", "UNR 121", "ARI 171","BAS 216"], 
    "S2": ["BAS 218"], 
    "5": ["ECE 234", "ECE 223", "CSE 251", "CSE 221", "BAS 217", "ARI 271","ENG 312","ECE 332"], 
    "6": ["ECE 224", "CSE 212", "ECE 235", "CSE 317", "ARI 381","UNR 471","CSE 351"], 
    "S3": ["Elective 1"], 
    "7": ["CSE 311", "CSE 313", "ECE 333","CSE 315", "ECE 321", "ARI 371","ARI 481","Elective 2"], 
    "8": ["Elective 3", "Elective 4", "CSE 423", "CSE 451", "CSE 452", "Elective 5", "ARI 482"], 
    "S4": [], 
    "9": [], 
    "10": [] 
}
PLAN_SLOTS = ["1", "2", "3", "4", "S2", "5", "6", "S3", "7", "8", "S4", "9", "10"]

class GraduationPlanner:
    def __init__(self, db_path):
        setup_course_db(db_path)
        with open(db_path, 'r', encoding='utf-8') as f:
            self.db = json.load(f)
        self.course_db = self.db['courses']
        self.plan = {slot: [] for slot in PLAN_SLOTS}
        self.all_courses_in_plan = set()

    def load_plan(self, plan_dict):
        self.plan = {slot: [] for slot in PLAN_SLOTS}
        for key, courses in plan_dict.items():
            if str(key) in self.plan:
                self.plan[str(key)] = courses
        self.all_courses_in_plan = {course for sem in self.plan.values() for course in sem}

    def get_course_info(self, course_code):
        return self.course_db.get(course_code)

    def validate_plan(self):
        courses_passed_so_far = set()
        all_errors = []
        gpa_warning = False
        
        for semester in PLAN_SLOTS:
            semester_courses = self.plan[semester]
            semester_credits = 0
            is_summer = semester.startswith("S")
            
            for course_code in semester_courses:
                course = self.get_course_info(course_code)
                if not course: continue
                course_name = course.get('name', course_code)
                semester_credits += course['credits']

                for pre in course.get('prerequisites', []):
                    if pre not in courses_passed_so_far:
                        all_errors.append(f"الفصل {semester}: {course_name} ({course_code}) 🚫 (المتطلب: {pre} لم يتم اجتيازه)")
                
                if "Cannot be taken in Summer" in course.get('constraints', []) and is_summer:
                     all_errors.append(f"الفصل {semester}: {course_name} ({course_code}) 🚫 (لا يمكن أخذها في الصيف)")

            max_credits = 9 if is_summer else 21
            if semester_credits > 18 and not is_summer: gpa_warning = True
            if semester_credits > max_credits:
                 all_errors.append(f"الفصل {semester}: 🚫 العبء ({semester_credits} س) يتجاوز الحد ({max_credits}).")

            courses_passed_so_far.update(semester_courses)

        return {
            "is_valid": len(all_errors) == 0,
            "errors": all_errors,
            "gpa_warning": gpa_warning
        }

    # دالة جديدة لتجهيز البيانات للواجهة
    def get_full_plan_details(self, plan_dict):
        self.load_plan(plan_dict)
        validation = self.validate_plan()
        
        detailed_plan = {}
        for sem, courses in self.plan.items():
            sem_details = []
            total_credits = 0
            for code in courses:
                info = self.get_course_info(code)
                if info:
                    sem_details.append({"code": code, "name": info["name"], "credits": info["credits"]})
                    total_credits += info["credits"]
            detailed_plan[sem] = {"courses": sem_details, "total_credits": total_credits}
            
        return {
            "validation": validation,
            "plan": detailed_plan
        }
    def move_courses_list(self, courses_to_move_dict, to_semester):
        """
        دالة نقل المواد من فصل دراسي لآخر
        """
        move_log = []
        for from_semester, courses in courses_to_move_dict.items():
            for course_code in courses:
                if from_semester == to_semester: 
                    continue
                if course_code in self.plan[from_semester]:
                    self.plan[from_semester].remove(course_code)
                if course_code not in self.plan[to_semester]:
                    self.plan[to_semester].append(course_code)
                move_log.append(f"تم نقل {course_code} من {from_semester} إلى {to_semester}")
                
        return "\n".join(move_log) if move_log else "لم يتم تحديد أي مواد للنقل."