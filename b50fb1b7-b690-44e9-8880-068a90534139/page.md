# LangChain и Langfuse: доклад

<!-- tab: Tab 1 -->

# LangChain и Langfuse: доклад

2026-11-07 · @someone

## Введение

LangChain — это Python-фреймворк, который собирает приложение на LLM из готовых блоков: модель, промпт, инструменты, база знаний. Langfuse — сервис, который хранит и версионирует промпты и показывает, что модель делала на каждом шаге. Вместе они закрывают путь от прототипа до работающего продукта.

План выступления (15–20 минут):

1. Зачем нужен фреймворк поверх API модели
2. Что такое LangChain и из чего он состоит
3. Живой пример кода: промпт → модель → ответ, затем агент с инструментом
4. Где LangChain применяют и когда он лишний
5. Langfuse: управление промптами и трассировка
6. Связка LangChain + Langfuse в коде
7. Итоги и вопросы

Проблема, с которой стоит начать: вызвать модель через API — 5 строк кода. Но в реальном приложении нужно подставлять данные в промпт, разбирать ответ, вызывать внешние функции, искать по документам, менять модель и понимать, почему ответ вышел плохим. Без общей структуры это быстро превращается в хаос.

## Что такое LangChain

LangChain — open-source библиотека (Python и JavaScript) для разработки приложений на больших языковых моделях. Её создал Харрисон Чейз в октябре 2022 года; версия 1.0 вышла осенью 2025 года. Главная идея — единый интерфейс ко всем моделям и сервисам: код не меняется, если заменить OpenAI на Claude, GigaChat или локальную модель через Ollama.

Экосистема состоит из нескольких пакетов:

| Пакет / продукт | Что делает |
| --- | --- |
| `langchain-core` | Базовые абстракции: сообщения, промпты, Runnable, инструменты |
| `langchain` | Готовые компоненты и функция `create_agent` |
| `langchain-openai`, `langchain-anthropic`, `langchain-ollama` и др. | Интеграции с конкретными провайдерами |
| `langchain-community` | Сотни интеграций от сообщества: загрузчики документов, векторные БД |
| LangGraph | Агенты и сложные сценарии как граф состояний, с памятью и циклами |
| LangSmith | Платный сервис мониторинга от авторов LangChain (аналог Langfuse) |

Аналогия для слушателей: если API модели — это двигатель, то LangChain — конструктор автомобиля с стандартными деталями, а Langfuse — приборная панель.

## Ключевые понятия LangChain

Всё в LangChain строится из восьми строительных блоков. У всех одинаковый интерфейс Runnable: методы `invoke`, `batch`, `stream` и их асинхронные версии.

| Блок | Зачем нужен | Класс / функция |
| --- | --- | --- |
| Модель (Chat Model) | Вызов LLM через единый интерфейс | `init_chat_model`, `ChatOpenAI`, `ChatAnthropic` |
| Сообщения | Роли диалога: system, user, assistant, tool | `SystemMessage`, `HumanMessage`, `AIMessage` |
| Шаблон промпта | Промпт с переменными `{topic}` | `ChatPromptTemplate` |
| Парсер ответа | Превращает текст в строку, JSON или Pydantic-объект | `StrOutputParser`, `with_structured_output` |
| Цепочка (LCEL) | Соединяет блоки оператором `\|` | `prompt \| model \| parser` |
| Инструмент (Tool) | Python-функция, которую модель может вызвать | декоратор `@tool` |
| Агент | Модель сама решает, какие инструменты и в каком порядке вызвать | `create_agent` |
| RAG (поиск по документам) | Ответ на основе ваших данных: загрузить → разрезать → эмбеддинги → векторная БД → поиск | `DocumentLoader`, `TextSplitter`, `VectorStore`, `Retriever` |

RAG (Retrieval-Augmented Generation) стоит объяснить отдельно: модель не знает ваших внутренних документов. Поэтому перед вопросом в промпт подкладывают 3–5 самых похожих кусков текста, найденных в векторной базе.

## Как использовать LangChain

Три примера по возрастанию сложности: цепочка, агент с инструментом и RAG. Код написан под LangChain 1.x.

### Установка

```bash
pip install -U langchain langchain-openai langchain-text-splitters
# ключ доступа к модели
export OPENAI_API_KEY="sk-..."
```

Без платного API можно взять локальную модель: `pip install langchain-ollama` и `init_chat_model("ollama:llama3.1")`. Остальной код не меняется.

### Пример 1. Цепочка: промпт → модель → парсер

```python
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = init_chat_model("openai:gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты преподаватель Python. Объясняй коротко."),
    ("user", "Объясни тему: {topic}"),
])

chain = prompt | model | StrOutputParser()   # LCEL: блоки соединены через |

print(chain.invoke({"topic": "декораторы"}))

# потоковый вывод — текст появляется по мере генерации
for chunk in chain.stream({"topic": "генераторы"}):
    print(chunk, end="", flush=True)
```

### Пример 2. Структурированный ответ (Pydantic)

```python
from pydantic import BaseModel, Field

class Review(BaseModel):
    sentiment: str = Field(description="positive / negative / neutral")
    score: int = Field(description="оценка от 1 до 5")

structured = model.with_structured_output(Review)
result = structured.invoke("Доставка быстрая, но коробка помятая")
print(result.sentiment, result.score)   # обычный Python-объект
```

### Пример 3. Агент с инструментом

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Вернуть погоду в городе."""
    return f"В городе {city} +5 °C, облачно"   # здесь мог бы быть запрос к API

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    system_prompt="Ты полезный ассистент.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "Какая погода в Казани?"}]})
print(result["messages"][-1].content)
```

Что происходит внутри: модель читает docstring инструмента, решает вызвать `get_weather("Казань")`, получает результат и формулирует ответ. Цикл «подумать → вызвать инструмент → посмотреть результат» повторяется, пока модель не даст финальный ответ.

### Пример 4. RAG: ответ по своим документам

```python
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = open("rules.txt", encoding="utf-8").read()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)

store = InMemoryVectorStore.from_texts(chunks, OpenAIEmbeddings())
retriever = store.as_retriever(search_kwargs={"k": 3})

question = "Сколько дней отпуска положено?"
context = "\n\n".join(doc.page_content for doc in retriever.invoke(question))

answer = model.invoke(f"Ответь только по контексту.\n\nКонтекст:\n{context}\n\nВопрос: {question}")
print(answer.content)
```

В реальном проекте `InMemoryVectorStore` заменяют на Chroma, FAISS, Qdrant или pgvector — интерфейс остаётся тем же.

## Где применим LangChain

LangChain оправдан там, где модель нужно связать с данными и другими системами. Типичные сценарии:

| Сценарий | Пример | Что используется |
| --- | --- | --- |
| Чат-бот по базе знаний | Поддержка отвечает по документации и FAQ | RAG, векторная БД |
| Ассистент для сотрудников | Поиск по регламентам, договорам, Confluence | RAG, загрузчики документов |
| Извлечение данных | Разбор резюме, счетов, отзывов в JSON | `with_structured_output` |
| Агенты | Бот сам ищет в интернете, делает SQL-запрос, создаёт задачу в Jira | `create_agent`, LangGraph, инструменты |
| Вопросы к базе данных | «Сколько заказов было в октябре?» → SQL → ответ | SQL-инструменты |
| Обработка текста пачками | Суммаризация, перевод, классификация тысяч обращений | Цепочки, `batch` |

**Плюсы:** быстрый прототип, смена модели одной строкой, сотни готовых интеграций, большое сообщество.

**Минусы и когда он не нужен:**

- Один вызов модели без инструментов проще сделать напрямую через SDK провайдера.
- Лишние слои абстракции усложняют отладку — поэтому и нужен Langfuse или LangSmith.
- API быстро менялся: статьи и ответы на Stack Overflow до 2025 года часто устарели (`LLMChain`, `initialize_agent` уже не используют).

Альтернативы, которые можно упомянуть: LlamaIndex (фокус на RAG), Haystack, Semantic Kernel от Microsoft, PydanticAI.

## Что такое Langfuse

Langfuse — open-source платформа для инженерии LLM-приложений: хранит промпты, записывает каждый вызов модели и помогает оценивать качество ответов. Его можно использовать в облаке (cloud.langfuse.com, есть бесплатный тариф) или развернуть у себя через Docker — это важно, когда данные нельзя отдавать наружу.

Зачем он нужен. В обычном проекте промпт лежит строкой в коде. Чтобы поправить одно слово, нужен коммит и деплой; непонятно, какая версия работала лучше и сколько стоил запрос. Langfuse выносит промпты из кода и делает работу модели видимой.

| Возможность | Что даёт |
| --- | --- |
| Prompt Management | Промпты хранятся в сервисе: версии, метки `production` / `staging`, переменные, откат в один клик |
| Tracing (трассировка) | Дерево всех шагов запроса: промпт, ответ, вызовы инструментов, время, токены |
| Стоимость и метрики | Расход в $ и задержка по моделям, пользователям, версиям промпта |
| Оценки (Evaluations) | Лайки пользователей, ручная разметка, LLM-as-a-judge |
| Datasets и эксперименты | Прогнать новую версию промпта по набору тестовых вопросов и сравнить со старой |
| Playground | Попробовать промпт на разных моделях прямо в браузере |

Ключевая мысль для слайда: промпт — это тоже код, и ему нужны версии, тесты и мониторинг. При этом менять его может не только программист, но и продакт-менеджер или эксперт предметной области — без деплоя.

## Управление промптами в Langfuse и связка с LangChain

Рабочий цикл выглядит так: промпт создают в Langfuse → приложение забирает его по имени и метке `production` → каждый вызов записывается в трейс → по метрикам решают, какую версию оставить. Новая версия промпта вступает в силу без деплоя — достаточно перенести на неё метку.

[Embedded widget: цикл работы с промптом · 4 шага](node/f3cd6671-a3d0)

Код приложения не меняется: меняется только версия промпта с меткой production.

### Установка и ключи

```bash
pip install -U langfuse langchain langchain-openai
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"   # или адрес своего сервера
```

Ключи берутся в настройках проекта в интерфейсе Langfuse.

### Шаг 1. Создать промпт

Обычно это делают в веб-интерфейсе, но можно и из кода. Переменные в Langfuse пишутся в двойных фигурных скобках.

```python
from langfuse import get_client

langfuse = get_client()

langfuse.create_prompt(
    name="python-tutor",
    type="chat",
    prompt=[
        {"role": "system", "content": "Ты преподаватель Python. Объясняй для уровня {{level}}."},
        {"role": "user", "content": "Объясни тему: {{topic}}"},
    ],
    labels=["production"],          # эта версия сразу боевая
    config={"model": "gpt-4o-mini", "temperature": 0},
)
```

Повторный вызов с тем же именем создаёт версию 2, 3 и т.д.; старые версии сохраняются.

### Шаг 2. Использовать промпт в LangChain и включить трассировку

```python
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

langfuse = get_client()

# 1. забрать актуальную боевую версию (кэшируется на клиенте)
lf_prompt = langfuse.get_prompt("python-tutor", type="chat", label="production")

# 2. превратить в шаблон LangChain: {{topic}} → {topic}
prompt = ChatPromptTemplate.from_messages(lf_prompt.get_langchain_prompt())
prompt.metadata = {"langfuse_prompt": lf_prompt}   # связать трейс с версией промпта

model = init_chat_model("openai:" + lf_prompt.config["model"],
                        temperature=lf_prompt.config["temperature"])
chain = prompt | model | StrOutputParser()

# 3. каждый вызов попадёт в Langfuse как трейс
handler = CallbackHandler()
answer = chain.invoke(
    {"level": "начинающий", "topic": "списковые включения"},
    config={"callbacks": [handler]},
)
print(answer)

langfuse.flush()   # в коротких скриптах: дождаться отправки данных
```

Тот же `CallbackHandler` работает и с агентом: `agent.invoke({...}, config={"callbacks": [handler]})` — в трейсе видно каждый вызов инструмента.

### Что показать на демо

1. Запустить скрипт и открыть трейс в Langfuse: промпт, ответ, токены, стоимость, время.
2. В интерфейсе поменять текст промпта (например, «отвечай с примером кода») и сохранить как версию 2 с меткой `production`.
3. Запустить тот же скрипт без изменений — ответ уже другой.
4. Показать сравнение метрик версий 1 и 2 и откат на версию 1.

## Итоги и план слайдов

LangChain отвечает на вопрос «как собрать LLM-приложение», Langfuse — на вопрос «как понять, что оно делает, и улучшать промпты без деплоя». Вместе они дают полный цикл: разработка → наблюдение → улучшение.

| № | Слайд | Содержание | Время, мин |
| --- | --- | --- | --- |
| 1 | Титул | Тема, автор | 0,5 |
| 2 | Проблема | Почему голого API мало | 1,5 |
| 3 | Что такое LangChain | Идея, история, экосистема пакетов | 2 |
| 4 | Строительные блоки | Модель, промпт, парсер, LCEL, инструменты, агент, RAG | 2 |
| 5 | Код: цепочка | Пример 1 + живой запуск | 2 |
| 6 | Код: агент | Пример 3, цикл «подумать → инструмент → ответ» | 2 |
| 7 | Где применяют | Таблица сценариев, плюсы и минусы | 1,5 |
| 8 | Что такое Langfuse | Промпт как код, возможности | 2 |
| 9 | LangChain + Langfuse | Код шага 2 | 2 |
| 10 | Демо | Смена промпта без деплоя, трейс | 3 |
| 11 | Итоги и вопросы | Три главные мысли | 1,5 |

**Три мысли для финального слайда:**

1. LangChain даёт единый интерфейс к любым моделям и готовые блоки для RAG и агентов.
2. Для простого одиночного вызова модели фреймворк не нужен.
3. Langfuse делает промпты управляемыми (версии, откат), а работу модели — видимой (трейсы, стоимость).

**Вопросы, которые могут задать, и короткие ответы:**

- *Чем Langfuse отличается от LangSmith?* — Langfuse открытый и разворачивается у себя бесплатно; LangSmith — закрытый сервис от авторов LangChain.
- *Можно ли Langfuse без LangChain?* — Да, есть SDK для Python/JS и интеграции с OpenAI SDK, LlamaIndex и другими.
- *Что будет, если Langfuse недоступен?* — SDK кэширует промпты, а в `get_prompt` можно передать `fallback` — запасной текст промпта.
- *LangChain или LangGraph?* — Для типового агента хватает `create_agent` (он построен на LangGraph); сам LangGraph нужен для нестандартных сценариев с ветвлениями.

## Источники

- [Документация LangChain](https://docs.langchain.com/)
- [Документация Langfuse](https://langfuse.com/docs)
- [Langfuse: Prompt Management](https://langfuse.com/docs/prompt-management/overview)
- [Langfuse: интеграция с LangChain](https://langfuse.com/integrations/frameworks/langchain)

Даты и API приведены по памяти, без проверки в интернете; перед выступлением стоит прогнать примеры с актуальными версиями пакетов.

