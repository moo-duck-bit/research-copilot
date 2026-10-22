COMPOSE = docker compose --env-file .env -f docker/docker-compose.yml
# web 서비스는 frontend 프로필에 묶여 있어서, 전체를 띄우거나 내릴 때는 프로필을 같이 준다.
FULL    = $(COMPOSE) --profile frontend

.PHONY: up up-api down ps logs test eval label verify-brief verify-analyze

up:            ## db + api + web 전체 빌드·기동 (브라우저: localhost:3000)
	$(FULL) up -d --build

up-api:        ## db + api만 기동 (웹 없이)
	$(COMPOSE) up -d --build db api

down:          ## 전체 정리 (web 포함, 데이터 볼륨은 보존)
	$(FULL) down

ps:
	$(FULL) ps

logs:
	$(COMPOSE) logs -f api

test:          ## pytest (호스트 포트를 열지 않는 일회성 api 컨테이너, db는 자동 기동)
	$(COMPOSE) run --rm --no-TTY api pytest

eval:          ## 판정 품질 평가 (label-only F1, selection+label F1, 혼동행렬)
	$(COMPOSE) up -d db api
	$(COMPOSE) exec -T api python eval/run_eval.py

label:         ## 골드셋 라벨링 CLI (대화형)
	$(COMPOSE) up -d db api
	# eval/는 호스트 폴더를 마운트한 것이라, 골드셋을 쓰려면 호스트 사용자로 실행한다.
	$(COMPOSE) exec --user $$(id -u):$$(id -g) api python eval/label_cli.py

verify-brief:  ## set_project_brief 3개 도메인 검증 (실제 Claude 호출)
	$(COMPOSE) exec -T api python scripts/verify_briefs.py

verify-analyze: ## analyze_content 5개 자료 검증 (실제 Claude/Voyage 호출)
	$(COMPOSE) exec -T api python scripts/verify_analyze.py
