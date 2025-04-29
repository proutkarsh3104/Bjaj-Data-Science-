import cv2
import pytesseract
import pandas as pd
import numpy as np
import io
from PIL import Image
import re
import logging
from typing import List, Dict, Tuple, Optional

try:
    from utils import is_out_of_range, clean_text
except ImportError:
    logging.error("Could not import from utils.py. Make sure it exists and is in the same directory.")
    def clean_text(text): return str(text).strip() if text else text
    def is_out_of_range(v, r): return False

TESSERACT_PATH = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    tesseract_version = pytesseract.get_tesseract_version()
    logging.info(f"Tesseract version {tesseract_version} found at: {TESSERACT_PATH}")
except pytesseract.TesseractNotFoundError:
    logging.error(f"Tesseract executable NOT FOUND at '{TESSERACT_PATH}'. Please install Tesseract and/or set the correct path.")
    raise RuntimeError(f"Tesseract not found at the specified path: {TESSERACT_PATH}")
except Exception as e:
    logging.error(f"An unexpected error occurred while configuring Tesseract: {e}")
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

MIN_OCR_CONFIDENCE = 40
LINE_VERTICAL_TOLERANCE_FACTOR = 0.7
MIN_COLUMN_GAP_THRESHOLD = 20
HEADER_KEYWORDS = {
    'test': ['test', 'investigation', 'parameter', 'analyte', 'description'],
    'value': ['result', 'value', 'observed', 'reading', 'observed value', 'results'],
    'unit': ['unit', 'units'],
    'range': ['range', 'reference', 'interval', 'normal', 'biological ref', 'biological ref interval', 'ref.', 'ref range', 'reference value']
}
HEADER_SEARCH_RATIO_LIMIT = 0.6

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    log.debug("Preprocessing image...")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        img_cv = np.array(image)

        if len(img_cv.shape) == 3:
            if img_cv.shape[2] == 4: img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)
            elif img_cv.shape[2] == 3: img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        elif len(img_cv.shape) == 2: img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        else: raise ValueError(f"Unsupported image shape: {img_cv.shape}")

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        log.info("Image preprocessed successfully.")
        return thresh
    except Exception as e:
        log.error(f"Error during image preprocessing: {e}", exc_info=True)
        raise ValueError(f"Failed to preprocess image: {e}")

def perform_ocr(image_np: np.ndarray) -> pd.DataFrame:
    log.debug("Performing OCR...")
    try:
        custom_config = r'--oem 3 --psm 6'
        ocr_data = pytesseract.image_to_data(
            image_np,
            output_type=pytesseract.Output.DATAFRAME,
            config=custom_config,
            lang='eng'
        )
        ocr_data = ocr_data[ocr_data.conf > MIN_OCR_CONFIDENCE]
        ocr_data = ocr_data.dropna(subset=['text'])
        ocr_data['text'] = ocr_data['text'].astype(str).str.strip()
        ocr_data = ocr_data[ocr_data['text'] != '']

        if ocr_data.empty:
            log.warning(f"OCR found no words with confidence > {MIN_OCR_CONFIDENCE}.")
            return pd.DataFrame()

        ocr_data['center_y'] = ocr_data['top'] + ocr_data['height'] / 2
        ocr_data['center_x'] = ocr_data['left'] + ocr_data['width'] / 2
        ocr_data['right'] = ocr_data['left'] + ocr_data['width']
        log.info(f"OCR performed. Found {len(ocr_data)} words meeting confidence threshold.")
        return ocr_data
    except Exception as e:
        log.error(f"Error during Tesseract OCR: {e}", exc_info=True)
        if "TesseractNotFoundError" in str(e) or "tesseract is not installed" in str(e).lower():
             log.critical("Tesseract executable not found or not in PATH.")
             raise RuntimeError(f"Tesseract not found: {e}")
        else:
            raise RuntimeError(f"Failed to perform OCR with Tesseract: {e}")

def group_words_into_lines(ocr_df: pd.DataFrame) -> List[List[Dict]]:
    log.debug("Grouping words into lines...")
    if ocr_df.empty:
        log.info("OCR DataFrame is empty, cannot group lines.")
        return []

    lines = []
    processed_indices = set()
    ocr_df = ocr_df.sort_values(by=['top', 'left']).reset_index(drop=True)

    for i in range(len(ocr_df)):
        if i in processed_indices: continue

        word = ocr_df.iloc[i]
        current_line_words = [word.to_dict()]
        processed_indices.add(i)
        word_center_y = word['center_y']
        word_height = word['height']
        vertical_tolerance = max(word_height * LINE_VERTICAL_TOLERANCE_FACTOR, 5)

        for j in range(i + 1, len(ocr_df)):
            if j in processed_indices: continue
            next_word = ocr_df.iloc[j]
            if abs(next_word['center_y'] - word_center_y) < vertical_tolerance:
                if next_word['left'] > word['left'] - (word['width'] / 2):
                    current_line_words.append(next_word.to_dict())
                    processed_indices.add(j)
            elif next_word['top'] > word['top'] + word_height * 1.5:
                break

        current_line_words.sort(key=lambda w: w['left'])
        lines.append(current_line_words)

    log.info(f"Grouped {len(ocr_df)} words into {len(lines)} lines.")
    return lines

def identify_columns(lines: List[List[Dict]]) -> Tuple[Optional[Dict[str, int]], Optional[int]]:
    log.debug(f"Identifying columns based on keywords within top {HEADER_SEARCH_RATIO_LIMIT*100:.0f}% of lines...")
    potential_headers = []
    search_limit_line_index = int(len(lines) * HEADER_SEARCH_RATIO_LIMIT)

    for i, line in enumerate(lines):
        if i > search_limit_line_index:
             log.info(f"Stopped searching for header row after line {i} ({HEADER_SEARCH_RATIO_LIMIT*100:.0f}% threshold).")
             break

        meaningful_words = [w['text'].lower() for w in line if len(w['text']) > 1]
        line_text = ' '.join(meaningful_words)
        log.debug(f"Line {i} Text for Header Check: '{line_text}'")
        found_headers_on_line = {}

        for col_type, keywords in HEADER_KEYWORDS.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', line_text):
                    for word in line:
                        if keyword in word['text'].lower():
                            if col_type not in found_headers_on_line:
                                found_headers_on_line[col_type] = word['left']
                                log.debug(f"  Found keyword '{keyword}' for type '{col_type}' at left={word['left']}")
                                break

        required_found = 'test' in found_headers_on_line
        other_found = any(k in found_headers_on_line for k in ['value', 'range'])

        if required_found and other_found:
             is_test_leftmost = all(found_headers_on_line['test'] <= pos for ct, pos in found_headers_on_line.items() if ct != 'test')
             if is_test_leftmost:
                 potential_headers.append({'index': i, 'columns': found_headers_on_line})
                 log.info(f"Potential header candidate found on line {i}: {found_headers_on_line}")

    if not potential_headers:
        log.warning("Column identification failed: No plausible header lines found matching keywords.")
        return None, None

    best_header = potential_headers[0]
    log.info(f"Selected header line {best_header['index']}. Final Column Positions: {best_header['columns']}")
    return best_header['columns'], best_header['index']

def extract_data_based_on_columns(lines: List[List[Dict]], column_starts: Dict[str, int], header_line_idx: int) -> List[Dict]:
    log.debug(f"Extracting data based on columns starting after line {header_line_idx}.")
    extracted_results = []
    if not column_starts: return []

    sorted_cols = sorted(column_starts.items(), key=lambda item: item[1])
    col_names_ordered = [item[0] for item in sorted_cols]
    col_lefts_ordered = [item[1] for item in sorted_cols]

    for i, line in enumerate(lines):
        if i <= header_line_idx: continue

        line_text_full = " ".join([w['text'] for w in line])
        is_likely_header = line_text_full.isupper() and len(line) < 4
        is_sparse = len(line) < 2
        if is_likely_header or is_sparse:
            log.debug(f"[Column Based] Skipping likely non-data line {i}: '{line_text_full}'")
            continue

        row_data = {col_name: [] for col_name in col_names_ordered}
        for word in line:
            word_center_x = word['left'] + word['width'] / 2
            assigned_col = None
            for j in range(len(col_names_ordered)):
                col_name = col_names_ordered[j]
                col_start = col_lefts_ordered[j]
                col_end = col_lefts_ordered[j+1] if j + 1 < len(col_lefts_ordered) else float('inf')
                tolerance = 10
                if col_start - tolerance <= word_center_x < col_end + tolerance:
                    assigned_col = col_name
                    break
            if assigned_col: row_data[assigned_col].append(word['text'])

        test_name_str = clean_text(" ".join(row_data.get('test', [])))
        test_value_str = clean_text(" ".join(row_data.get('value', [])))
        test_unit_str = clean_text(" ".join(row_data.get('unit', [])))
        range_str = clean_text(" ".join(row_data.get('range', [])))

        if test_name_str and (test_value_str or range_str):
            out_of_range = is_out_of_range(test_value_str, range_str)
            result = {
                "test_name": test_name_str,
                "test_value": test_value_str if test_value_str else None,
                "bio_reference_range": range_str if range_str else None,
                "test_unit": test_unit_str if test_unit_str else None,
                "lab_test_out_of_range": out_of_range
            }
            extracted_results.append(result)

    log.info(f"[Column Based] Extraction finished. Found {len(extracted_results)} results.")
    return extracted_results

def extract_data_heuristically(lines: List[List[Dict]]) -> List[Dict]:
    log.info("Executing heuristic data extraction (no headers found).")
    extracted_results = []
    estimated_start_y = lines[0][0]['top'] + 50 if lines and lines[0] else 0

    for i, line in enumerate(lines):
        if not line: continue
        if line[0]['top'] < estimated_start_y: continue
        if len(line) < 2: continue

        columns_in_line = []
        current_column_words = []
        last_word_right = 0

        for word in line:
            word_left = word['left']
            word_right = word['left'] + word['width']
            gap = word_left - last_word_right

            if current_column_words and gap > MIN_COLUMN_GAP_THRESHOLD:
                columns_in_line.append(" ".join(current_column_words))
                current_column_words = [word['text']]
            else:
                current_column_words.append(word['text'])
            last_word_right = word_right

        if current_column_words: columns_in_line.append(" ".join(current_column_words))

        if len(columns_in_line) >= 2:
            test_name_str = clean_text(columns_in_line[0])
            test_value_str = clean_text(columns_in_line[1])
            test_unit_str = None
            range_str = None

            if test_name_str and not re.fullmatch(r'[\d\.,\s<>-]+', test_name_str):

                if len(columns_in_line) > 2:
                    potential_range = clean_text(columns_in_line[2])
                    if re.search(r'\d\s*-\s*\d', potential_range) or re.search(r'[<>]\s*\d', potential_range):
                         range_str = potential_range
                         log.debug(f"[Heuristic] Tentatively identified range in column 3: '{range_str}'")

                unit_match = re.search(r'([a-zA-Z%µ\/]+[\d\^a-zA-Zµ\/]*)$', test_value_str)
                if unit_match:
                    test_unit_str = unit_match.group(1).strip()

                out_of_range = is_out_of_range(test_value_str, range_str)
                result = {
                    "test_name": test_name_str,
                    "test_value": test_value_str if test_value_str else None,
                    "bio_reference_range": range_str if range_str else None,
                    "test_unit": test_unit_str if test_unit_str else None,
                    "lab_test_out_of_range": out_of_range
                }
                extracted_results.append(result)

    log.info(f"[Heuristic] Extraction finished. Found {len(extracted_results)} results.")
    return extracted_results

def extract_lab_data_from_image(image_bytes: bytes) -> Dict:
    final_output = {"is_success": False, "data": []}
    log.info("Starting lab data extraction process...")
    try:
        preprocessed_image = preprocess_image(image_bytes)

        ocr_df = perform_ocr(preprocessed_image)
        if ocr_df.empty:
            log.error("OCR processing returned no data. Cannot proceed.")
            final_output["error"] = "OCR failed or no text detected with sufficient confidence."
            return final_output

        lines = group_words_into_lines(ocr_df)
        if not lines:
             log.error("Layout analysis failed: Could not group OCR words into lines.")
             final_output["error"] = "Layout analysis failed (line grouping)."
             return final_output

        column_positions, header_line_idx = identify_columns(lines)

        extracted_data = []
        if column_positions is not None and header_line_idx is not None:
            log.info(f"Header-based column identification successful (Header Line: {header_line_idx}).")
            extracted_data = extract_data_based_on_columns(lines, column_positions, header_line_idx)
        else:
            log.warning("Header-based column identification failed. Attempting heuristic extraction.")
            extracted_data = extract_data_heuristically(lines)
            if not extracted_data:
                 log.warning("Heuristic extraction did not yield any results.")

        final_output["data"] = extracted_data
        final_output["is_success"] = True
        log.info(f"Processing complete. Success: {final_output['is_success']}, Total Results found: {len(extracted_data)}")

    except (ValueError, RuntimeError) as e:
        log.error(f"Error during lab data extraction process: {e}", exc_info=True)
        final_output["is_success"] = False
        final_output["error"] = f"An internal error occurred: {e}"
    except Exception as e:
        log.error(f"Unexpected error during lab data extraction process: {e}", exc_info=True)
        final_output["is_success"] = False
        final_output["error"] = f"An unexpected internal error occurred: {e}"

    return final_output