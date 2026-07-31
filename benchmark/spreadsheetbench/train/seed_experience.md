# Spreadsheet Manipulation Skill (xlsx)

## Overview

This skill guides manipulating Excel (.xlsx) spreadsheets using Python.

**Primary libraries**: `openpyxl` (structure-preserving read/write), `pandas`
(data transformation). Never use any other third-party libraries.

## Common Workflow

1. **Explore** the input file: list sheets, inspect headers, check dimensions.
2. **Write** a script that reads the input xlsx and writes the output xlsx.
3. **Confirm** the target cells/range contain the expected values.

## Library Selection

| Use case | Library |
|----------|---------|
| Preserve formulas, formatting, named ranges | `openpyxl` |
| Bulk data transformation, aggregation, sorting | `pandas` → write back with `openpyxl` |
| Simple cell read/write | `openpyxl` |

**Warning**: `pandas.to_excel()` silently destroys existing formulas and named
ranges. When writing back to a spreadsheet that contains formulas, always use
`openpyxl.save()`.

## Output Requirements

- Save the result to the output path given in the task.
- Do not hardcode row counts or column letters — iterate over actual rows in the
  workbook.
- Preserve sheets and cells not mentioned in the instruction.
