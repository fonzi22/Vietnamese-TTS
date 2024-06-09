# Chọn ngôn ngữ:
language = "vi" 
input_text ="Xin chào, tôi là một công cụ có khả năng chuyển đổi văn bản thành giọng nói tự nhiên, được phát triển bởi nhóm Nón lá. Tôi có thể hổ trợ người khiếm thị,  đọc sách nói, làm trợ lý ảo, review phim, làm waifu để an ủi bạn, và phục vụ nhiều mục đích khác." # @param {type:"string"}
#  Tự động chuẩn hóa chữ (VD: 20/11 -> hai mươi tháng mười một)
normalize_text = True 
# In chi tiết xử lý
verbose = True 

def cry_and_quit():
    print("> Lỗi rồi huhu 😭😭, bạn hãy nhấn chạy lại phần này nhé!")
    quit()

import os
import string
import unicodedata
from datetime import datetime
from pprint import pprint

import torch
import torchaudio
from tqdm import tqdm
from underthesea import sent_tokenize, word_tokenize
from unidecode import unidecode

try:
    from vinorm import TTSnorm
    from TTS.TTS.tts.configs.xtts_config import XttsConfig
    from TTS.TTS.tts.models.xtts import Xtts
except:
    cry_and_quit()

# Load model
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "You need to run the previous steps or manually set the `XTTS checkpoint path`, `XTTS config path`, and `XTTS vocab path` fields !!"
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("Loading XTTS model! ")
    XTTS_MODEL.load_checkpoint(config,
                               checkpoint_path=xtts_checkpoint,
                               vocab_path=xtts_vocab,
                               use_deepspeed=True)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("Model Loaded!")
    return XTTS_MODEL


def get_file_name(text, max_char=50):
    filename = text[:max_char]
    filename = filename.lower()
    filename = filename.replace(" ", "_")
    filename = filename.translate(str.maketrans("", "", string.punctuation.replace("_", "")))
    filename = unidecode(filename)
    current_datetime = datetime.now().strftime("%m%d%H%M%S")
    filename = f"{current_datetime}_{filename}"
    return filename


def calculate_keep_len(text, lang):
    if lang in ["ja", "zh-cn"]:
        return -1

    word_count = len(text.split())
    num_punct = (
        text.count(".")
        + text.count("!")
        + text.count("?")
        + text.count(",")
    )

    if word_count < 5:
        return 15000 * word_count + 2000 * num_punct
    elif word_count < 10:
        return 13000 * word_count + 2000 * num_punct
    return -1


def normalize_vietnamese_text(text):
    text = (
        TTSnorm(text, unknown=False, lower=False, rule=True)
        .replace("-", "")
        .replace("...", ".")
        .replace("..", ".")
        .replace(":.", ".")
        .replace("!.", "!")
        .replace("?.", "?")
        .replace(" .", ".")
        .replace(" ,", ",")
        .replace('"', "")
        .replace("'", "")
        .replace("  ", " ")
        .replace("AI", "Ây Ai")
        .replace("A.I", "Ây Ai")
    )
    return text

def merge_short_sentences(text):
    sentences = sent_tokenize(text)
    merged_sentences = []
    i = 0

    while i < len(sentences):
        words = word_tokenize(sentences[i])
        # Kiểm tra số lượng từ trong câu
        if len(words) < 10 and i+1 < len(sentences):
            # Gộp câu ngắn với câu kế tiếp
            sentences[i+1] = sentences[i][:-1] + ', ' + sentences[i+1]
        else:
            # Nếu câu không ngắn, thêm nó vào danh sách các câu đã gộp
            merged_sentences.append(sentences[i])
        i += 1

    return merged_sentences


def run_tts(XTTS_MODEL, lang, tts_text, speaker_audio_file,
            normalize_text= True,
            verbose=False):
    """
    Run text-to-speech (TTS) synthesis using the provided XTTS_MODEL.

    Args:
        XTTS_MODEL: A pre-trained TTS model.
        lang (str): The language of the input text.
        tts_text (str): The text to be synthesized into speech.
        speaker_audio_file (str): Path to the audio file of the speaker to condition the synthesis on.
        normalize_text (bool, optional): Whether to normalize the input text. Defaults to True.
        verbose (bool, optional): Whether to print verbose information. Defaults to False.
        output_chunks (bool, optional): Whether to save synthesized speech chunks separately. Defaults to False.

    Returns:
        str: Path to the synthesized audio file.
    """

    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to run the previous step to load the model !!", None, None

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        audio_path=speaker_audio_file,
        gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
        max_ref_length=XTTS_MODEL.config.max_ref_len,
        sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
    )

    if normalize_text and lang == "vi":
        # Bug on google colab
        try:
            tts_text = normalize_vietnamese_text(tts_text)
        except:
            cry_and_quit()

    if lang in ["ja", "zh-cn"]:
        tts_texts = tts_text.split("。")
    else:
        tts_texts = merge_short_sentences(tts_text)

    if verbose:
        print("Text for TTS:")
        pprint(tts_texts)

    wav_chunks = []
    for text in tqdm(tts_texts):
        if text.strip() == "":
            continue

        wav_chunk = XTTS_MODEL.inference(
            text=text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.3,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=30,
            top_p=0.85,
        )

        # Quick hack for short sentences
        keep_len = calculate_keep_len(text, lang)
        wav_chunk["wav"] = torch.tensor(wav_chunk["wav"][:keep_len])

        wav_chunks.append(wav_chunk["wav"])

    out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
    out_path = os.path.join(output_dir, f"{get_file_name(tts_text)}.wav")
    torchaudio.save(out_path, out_wav, 24000)

    if verbose:
        print(f"Saved final file to {out_path}")

    return out_path

print("> Đang nạp mô hình...")
vixtts_model = load_model(xtts_checkpoint="model/model.pth",
                        xtts_config="model/config.json",
                        xtts_vocab="model/vocab.json")
print("> Đã nạp mô hình")

while True:
  voice = int(input("Chọn giọng đọc [1, 2, 3]:"))
  if voice == 1:
    reference_audio = "voice/voice1.wav"
  elif voice == 2:
    reference_audio = "voice/voice2.wav"
  elif voice == 3:
    reference_audio = "voice/voice3.wav"
  else:
    print("Lựa chọn không phù hợp. Hãy chọn lại.")
    continue
  with open('input.txt', encoding='utf-8') as f:
    input_text = f.read()
  print("> Đang chuyển đổi...")
  audio_file = run_tts(vixtts_model,
          lang=language,
          tts_text=input_text,
          speaker_audio_file=reference_audio,
          normalize_text=normalize_text,
          verbose=verbose)
  print("> Đã chuyển đổi xong!")
  if input("Bạn có muốn tiếp tục không [1:Có, 0:Không]") != '1':
    break
