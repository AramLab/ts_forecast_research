# Makefile для запуска экспериментов по M3/M4 с безопасными настройками памяти.
# На macOS жёсткое ограничение памяти через ulimit ограниченно; лучший способ
# избежать крашей — уменьшить число worker-процессов и отключить тяжёлые модели.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

PYTHON ?= python
WORKERS ?= 2
N_SERIES ?= 150
TIMEOUT ?= 600
SEED ?= 42
CEEMDAN_TRIALS ?= 20
N_WAVELET_MODES ?= 2
WAVELET ?= db4
OUT_DIR ?= results
M3_TSF ?= m3/datasets/m3_monthly_dataset.tsf
M3_GROUP ?= Monthly
M4_GROUP ?= Monthly
LOW_MEMORY ?= 0

ifeq ($(LOW_MEMORY),1)
NO_TRANSFORMER := --no_transformer
NO_CEEMDAN := --no_ceemdan
NO_WAVELET := --no_wavelet
NO_LSTM := --no_lstm
else
NO_TRANSFORMER :=
NO_CEEMDAN :=
NO_WAVELET :=
NO_LSTM :=
endif

.PHONY: help run all m3-monthly m3-quarterly m3-yearly m3-other m4-monthly m4-quarterly m4-yearly m4-weekly m4-daily m4-hourly clean

help:
	@printf "Usage:\n"
	@printf "  make run DATASET=m3 M3_GROUP=Monthly\n"
	@printf "  make m3-quarterly\n"
	@printf "  make m4-yearly\n"
	@printf "  make all\n\n"
	@printf "Variables:\n"
	@printf "  WORKERS=%s\n" "$(WORKERS)"
	@printf "  N_SERIES=%s\n" "$(N_SERIES)"
	@printf "  LOW_MEMORY=%s (1=skip CEEMDAN/Wavelet/LSTM/Transformer)\n" "$(LOW_MEMORY)"
	@printf "  CEEMDAN_TRIALS=%s\n" "$(CEEMDAN_TRIALS)"
	@printf "  N_WAVELET_MODES=%s\n" "$(N_WAVELET_MODES)"
	@printf "  OUT_DIR=%s\n" "$(OUT_DIR)"

run:
ifndef DATASET
	$(error DATASET is required. Example: make run DATASET=m3 M3_GROUP=Monthly)
endif
	@echo "Running $(DATASET) $(if $(filter $(DATASET),m3),M3 group=$(M3_GROUP),M4 group=$(M4_GROUP))"
	@$(PYTHON) run_experiment.py \
		--dataset $(DATASET) \
		$(if $(filter $(DATASET),m3),--m3_group $(M3_GROUP),) \
		$(if $(filter $(DATASET),m4),--m4_group $(M4_GROUP),) \
		$(if $(filter $(DATASET),m3),--m3_tsf $(M3_TSF),) \
		--n_series $(N_SERIES) \
		--workers $(WORKERS) \
		--timeout $(TIMEOUT) \
		--seed $(SEED) \
		--ceemdan_trials $(CEEMDAN_TRIALS) \
		--n_wavelet_modes $(N_WAVELET_MODES) \
		--wavelet $(WAVELET) \
		$(NO_TRANSFORMER) $(NO_CEEMDAN) $(NO_WAVELET) $(NO_LSTM) \
		--out $(OUT_DIR)/$(DATASET)-$(if $(filter $(DATASET),m3),$(M3_GROUP),$(M4_GROUP))

all: m3-monthly m3-quarterly m3-yearly m3-other m4-monthly m4-quarterly m4-yearly
	@echo "All selected dataset groups finished."

m3-monthly:
	$(MAKE) run DATASET=m3 M3_GROUP=Monthly OUT_DIR=$(OUT_DIR)/m3-monthly

m3-quarterly:
	$(MAKE) run DATASET=m3 M3_GROUP=Quarterly OUT_DIR=$(OUT_DIR)/m3-quarterly

m3-yearly:
	$(MAKE) run DATASET=m3 M3_GROUP=Yearly OUT_DIR=$(OUT_DIR)/m3-yearly

m3-other:
	$(MAKE) run DATASET=m3 M3_GROUP=Other OUT_DIR=$(OUT_DIR)/m3-other

m4-monthly:
	$(MAKE) run DATASET=m4 M4_GROUP=Monthly OUT_DIR=$(OUT_DIR)/m4-monthly

m4-quarterly:
	$(MAKE) run DATASET=m4 M4_GROUP=Quarterly OUT_DIR=$(OUT_DIR)/m4-quarterly

m4-yearly:
	$(MAKE) run DATASET=m4 M4_GROUP=Yearly OUT_DIR=$(OUT_DIR)/m4-yearly

m4-weekly:
	$(MAKE) run DATASET=m4 M4_GROUP=Weekly OUT_DIR=$(OUT_DIR)/m4-weekly

m4-daily:
	$(MAKE) run DATASET=m4 M4_GROUP=Daily OUT_DIR=$(OUT_DIR)/m4-daily

m4-hourly:
	$(MAKE) run DATASET=m4 M4_GROUP=Hourly OUT_DIR=$(OUT_DIR)/m4-hourly

lowmem:
	@printf "Low-memory mode: workers=$(WORKERS), n_series=$(N_SERIES)\n"
	$(MAKE) run LOW_MEMORY=1

clean:
	rm -rf $(OUT_DIR)/m3-* $(OUT_DIR)/m4-* results/m3-* results/m4-*
