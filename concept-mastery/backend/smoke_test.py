"""Runs the whole loop without the web server:  python smoke_test.py"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from app.pipelines import ingest, tutor, quiz, mastery, recommend, loop

candidates = [
    Path(__file__).parents[1] / "data" / "uploads" / "ml-basics.pdf",
    Path(__file__).parents[1] / "sample" / "ml-basics.pdf",
    Path.cwd() / "sample" / "ml-basics.pdf",
]
pdf = next((p for p in candidates if p.exists()), None)
if not pdf:
    print("No sample PDF found. Exiting.")
    sys.exit(0)
print("INGEST   ", ingest.run_ingest(pdf))
print("TUTOR    ", tutor.ask("What happens if the learning rate is too high?"))
for i in range(8):
    q = quiz.next_question()
    ok = random.random() < 0.7
    out = loop.run(q["conceptId"], ok)
    print(f"QUIZ {i+1}  [{q['conceptName']}] {q['question'][:70]}...  correct={ok}  next={out['next_id']}")
for m in mastery.snapshot():
    print("MASTERY  ", m["name"], m["status"], m["score"], "-", m["why"])
for a in recommend.recommend()["actions"][:3]:
    print("NEXT     ", a["name"], "|", a["type"], "| pages", a["pages"])
