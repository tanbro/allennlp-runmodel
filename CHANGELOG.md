# CHANGELOG

## 0.2.1

Date: 2018-12-24

- Change:
  - `AllenNLP` 0.8
- New:
  - `TOML` logging configuration file format

## 0.2.0

Date: 2018-11-20

- New:
  - Multiple models!
- Change:
  - Rewrite arguments parsing part, based on `click`.
  - URL changed because of multiple models feature. `model={{model_name}}` query parameter needed.

Many other modifications.

## 0.1.2

Date: 2018-11-15

- Change:
  - Process worker mode by default.
  - Dynamic workers count by default.
- Add:
  - `--workers-type` argument.
- Adjust:
  - Many logging texts, and print to stdout by default.
  - Remove some useless variables.

And many other modifications.

## 0.1.1

Date: 2018-11-14

- Change:
  - Rename argument `--log-conf` to `--logging-config`
- Add:
  - `--num-threads` argument
- Modify:
  - Add `version` module.

And many other modifications.

## 0.1.0

Date: 2018-11-07

A very early experimental release
