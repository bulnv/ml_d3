# Анализ качества и очистка датасета мошеннических финансовых операций

## Описание проекта

Данный проект представляет собой решение домашнего задания №3 курса MLOps. Основная цель - анализ качества данных и разработка процесса очистки датасета финансовых транзакций с использованием Apache Spark.

## Структура данных

Исходный датасет содержит следующие поля:
- transaction_id: ID транзакции
- tx_datetime: Дата и время транзакции
- customer_id: ID клиента
- terminal_id: ID терминала
- tx_amount: Сумма транзакции
- tx_time_seconds: Время в секундах
- tx_time_days: Время в днях
- tx_fraud: Флаг мошеннической операции
- tx_fraud_scenario: Тип мошеннического сценария

## Выявленные проблемы с данными

1. **Дубликаты**:
   - Обнаружено 8 полных дубликатов транзакций
   - Затронуты транзакции с ID: 1862497332, 1862070073, 1871745328 и др.

2. **Пропущенные значения**:
   - terminal_id: 2299 пропущенных значений
   - Одна запись с полностью пустыми значениями

3. **Аномалии в числовых полях**:
   - tx_amount: 724,473 выбросов (1.54%)
   - tx_fraud: 1,406,151 аномальных значений (2.99%)
   - tx_fraud_scenario: 1,406,151 аномальных значений (2.99%)

## Реализованное решение

### Инфраструктура
- Создан кластер Spark на Yandex Cloud DataProc
- Настроено хранилище данных в Object Storage
- Реализована автоматизация развертывания через Terraform

### Процесс очистки данных
1. Удаление полных дубликатов
2. Обработка выбросов в числовых данных
3. Заполнение пропущенных значений
4. Валидация временных меток
5. Проверка согласованности fraud-меток

### Технические характеристики
- Размер исходного датасета: 46,998,984 записей
- Формат хранения: Parquet с партиционированием по дате
- Используемые технологии: Python, PySpark, AWS S3

## Настройка и запуск

### Зависимости
```bash
pip install -r requirements.txt
```

### Запуск очистки данных
```bash
export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_ENDPOINT_URL_S3=https://storage.yandexcloud.net/

spark-submit \
  --master yarn \
  --deploy-mode client \
  --num-executors 3 \
  --executor-memory 10G \
  --executor-cores 4 \
  --driver-memory 4G \
  --conf spark.executor.memoryOverhead=2G \
  --conf "spark.hadoop.fs.s3a.access.key=$AWS_ACCESS_KEY" \
  --conf "spark.hadoop.fs.s3a.secret.key=$AWS_SECRET_KEY" \
  --conf "spark.hadoop.fs.s3a.endpoint=storage.yandexcloud.net" \
  clean.py \
  /user/ubuntu/data/2022-11-04.txt \
  s3a://hw3-buck-b1g2eh9k7s7mmebes9li/cleaned_transactions/
  ```


## Результаты очистки
- Исходное количество записей: 46,998,984
- Количество записей после очистки: ~45,592,832
- Удалено аномальных записей: ~1,406,152 (2.99%)

## Структура репозитория
```
├── README.md
├── requirements.txt
├── clean_transactions.py
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── notebooks/
│   └── h3.ipynb

```

## Автор
Nikolai Bulashev

## Лицензия
MIT
