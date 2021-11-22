import ppocr.infer.utility as utility
from ppocr.infer.predict_system import TextSystem

args = utility.server_parse_args()
text_sys = TextSystem(args)
