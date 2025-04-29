import re
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

RANGE_PATTERN_NUMERIC = re.compile(r'^\s*([\d,\.]+)\s*-\s*([\d,\.]+)\s*$')
RANGE_PATTERN_LT = re.compile(r'^\s*(?:<|Less than)\s*([\d,\.]+)\s*$')
RANGE_PATTERN_GT = re.compile(r'^\s*(?:>|Greater than)\s*([\d,\.]+)\s*$')
RANGE_PATTERN_UPTO = re.compile(r'^\s*(?:up to|upto)\s*([\d,\.]+)\s*$')
VALUE_PATTERN = re.compile(r'([<>])?\s*([\d,\.]+)')

def clean_numeric_string(s):
    if not isinstance(s, str):
        return None
    try:
        cleaned = s.replace(',', '').strip()
        return float(cleaned)
    except ValueError:
        return None

def is_out_of_range(value_str: str | None, range_str: str | None) -> bool:
    if value_str is None or range_str is None:
        return False

    value_str_cleaned = str(value_str).strip()
    numeric_value = None
    value_match = VALUE_PATTERN.search(value_str_cleaned)

    if value_match:
        prefix, num_str = value_match.groups()
        numeric_value = clean_numeric_string(num_str)
        if numeric_value is None:
             log.debug(f"Could not convert value '{num_str}' to float.")
             return False
        if prefix == '<':
            numeric_value -= 0.00001
        elif prefix == '>':
             numeric_value += 0.00001
    else:
        log.debug(f"Value '{value_str_cleaned}' does not match expected numeric pattern.")
        return False

    range_str_cleaned = str(range_str).strip()
    log.debug(f"Checking value '{numeric_value}' against range '{range_str_cleaned}'")

    match_numeric = RANGE_PATTERN_NUMERIC.match(range_str_cleaned)
    match_lt = RANGE_PATTERN_LT.match(range_str_cleaned)
    match_gt = RANGE_PATTERN_GT.match(range_str_cleaned)
    match_upto = RANGE_PATTERN_UPTO.match(range_str_cleaned)

    try:
        if match_numeric:
            low_str, high_str = match_numeric.groups()
            low = clean_numeric_string(low_str)
            high = clean_numeric_string(high_str)
            if low is not None and high is not None:
                log.debug(f"Numeric Range Comparison: {low} <= {numeric_value} <= {high}?")
                return not (low <= numeric_value <= high)
            else:
                 log.warning(f"Could not parse numeric range bounds: '{low_str}', '{high_str}'")
        elif match_lt:
            limit_str = match_lt.group(1)
            limit = clean_numeric_string(limit_str)
            if limit is not None:
                log.debug(f"Less Than Range Comparison: {numeric_value} < {limit}?")
                return not (numeric_value < limit)
            else:
                log.warning(f"Could not parse LT range bound: '{limit_str}'")
        elif match_gt:
            limit_str = match_gt.group(1)
            limit = clean_numeric_string(limit_str)
            if limit is not None:
                log.debug(f"Greater Than Range Comparison: {numeric_value} > {limit}?")
                return not (numeric_value > limit)
            else:
                log.warning(f"Could not parse GT range bound: '{limit_str}'")
        elif match_upto:
            limit_str = match_upto.group(1)
            limit = clean_numeric_string(limit_str)
            if limit is not None:
                log.debug(f"Up To Range Comparison: {numeric_value} <= {limit}?")
                return not (numeric_value <= limit)
            else:
                 log.warning(f"Could not parse Up To range bound: '{limit_str}'")
        else:
            log.debug(f"Range string '{range_str_cleaned}' did not match numeric patterns.")
            return False
    except Exception as e:
        log.error(f"Error during range comparison for value '{value_str}' and range '{range_str}': {e}")
        return False

    log.debug(f"Comparison inconclusive for value '{value_str}' and range '{range_str}'.")
    return False

def clean_text(text):
    if isinstance(text, str):
        return text.strip()
    return text