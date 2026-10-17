"""Русские названия для таблиц и графиков. Имена входных столбцов не меняются.

Смысл и доступность признаков описаны в FEATURES.md. Здесь находятся только
подписи, чтобы одинаковый признак в разных ноутбуках назывался одинаково.
"""
import re
import textwrap
import pandas as pd
from IPython.display import display

GROUP_LABELS = {
    'identity': 'Исходные категории', 'calendar': 'Календарь и возраст пары',
    'history': 'История продаж пары', 'stock': 'Вчерашние остатки',
    'aggregate': 'Продажи склада и артикула', 'crosses': 'Явные сочетания категорий',
}
MODEL_LABELS = {
    'zero': 'Всегда ноль', 'seasonal_lag7': 'Продажи 7 дней назад',
    'rolling_mean7': 'Среднее прошлых 7 дней',
    'linear_base': 'Ridge: базовые поля', 'linear_fe': 'Ridge: с FE',
    'catboost_base': 'CatBoost: базовые поля', 'catboost_fe': 'CatBoost: с FE',
    'dcn_base': 'DCN: базовые поля', 'dcn_fe': 'DCN: с FE',
    'mlp_base': 'MLP: базовые поля', 'mlp_fe': 'MLP: с FE',
    'linear': 'Линейная регрессия (Ridge)', 'catboost': 'CatBoost',
    'dcn': 'Нейросеть DCN', 'mlp': 'Нейросеть без cross-ветки',
    'base': 'Базовые поля', 'fe': 'С добавленными признаками',
    'valid': 'Валидация', 'test': 'Тест', 'train': 'Обучение',
    'identity_only': 'Только исходные категории',
    'calendar_identity': '+ Календарь', 'history_aggregate': '+ История и агрегаты',
    'history_stock': '+ Остатки', 'full_crosses': '+ Явные сочетания',
    'full_crosses_mae': 'Полный набор, MAE-loss',
    'all': 'Все дни', 'zero_sales': 'Дни без продаж', 'positive_sales': 'Дни с продажами',
    'pair_age_lt28': 'Пары моложе 28 дней', 'stock_missing': 'Нет вчерашнего остатка',
}
LABELS = {
    'date': 'Дата', 'warehouse_name': 'Склад', 'article': 'Артикул',
    'composite_division': 'Подразделение', 'cities_name': 'Город',
    'category_lvl_1': 'Категория 1-го уровня', 'category_lvl_2': 'Категория 2-го уровня',
    'category_lvl_3': 'Категория 3-го уровня', 'characteristic': 'Характеристика товара',
    'quantity': 'Количество продаж', 'sales_sum': 'Сумма продаж: исходная шкала',
    'discount_sum': 'Скидка: исходная шкала', 'stock': 'Остаток',
    'stock_mean': 'Средний остаток на дневном ключе', 'stock_min': 'Минимальный остаток на ключе',
    'stock_max': 'Максимальный остаток на ключе', 'stock_records': 'Число записей остатка на ключе',
    'dow': 'День недели', 'month': 'Месяц', 'day_of_month': 'День месяца',
    'day_of_year': 'День года', 'week_of_year': 'Неделя года', 'year': 'Год',
    'is_weekend': 'Выходной день', 'dow_sin': 'День недели: синус', 'dow_cos': 'День недели: косинус',
    'month_sin': 'Месяц: синус', 'month_cos': 'Месяц: косинус',
    'pair_age_days': 'Возраст пары, дней', 'metadata_known': 'Описание товара уже известно',
    'history_missing': 'Нет наблюдения продаж за вчера', 'qty_trend_7_28': 'Отношение средних за 7 и 28 дней',
    'stock_missing_lag_1': 'Нет вчерашнего остатка',
    'stock_spread_lag_1': 'Вчерашний разброс остатков',
    'stock_nonpositive_lag_1': 'Вчера остаток неположительный',
    'stock_cover_7': 'Остаток к средним продажам за 7 дней',
    'sell_to_stock_lag_1': 'Вчерашние продажи к остатку',
    'article_warehouse': 'Артикул × склад', 'article_dow': 'Артикул × день недели',
    'warehouse_dow': 'Склад × день недели',
    'dtype': 'Тип в Python', 'unique': 'Разных значений', 'missing': 'Пропусков',
    'missing_pct': 'Пропусков, %', 'zero_pct': 'Нулей, %', 'negative_pct': 'Отрицательных, %',
    'family':'Семейство модели', 'feature_set':'Набор входов', 'reference':'Простое правило',
    'sales':'Продажи', 'sales_daily_keys':'Дневных ключей продаж',
    'sales_keys_without_stock_pct':'Ключей продаж без остатка, %',
    'sales_keys_with_nonpositive_stock_pct':'Ключей с неположительным остатком, %',
    'table': 'Таблица', 'column': 'Столбец', 'count': 'Число значений',
    'mean': 'Среднее', 'std': 'Стандартное отклонение', 'min': 'Минимум', 'max': 'Максимум',
    'rows': 'Число строк', 'dates': 'Дней', 'date_min': 'Первая дата', 'date_max': 'Последняя дата',
    'exact_duplicate_rows': 'Полных дубликатов строк',
    'extra_rows_over_daily_key': 'Лишних строк на дневном ключе',
    'daily_stock_keys': 'Дневных ключей остатков', 'repeated_keys': 'Ключей с повторами',
    'conflicting_keys': 'Ключей с разными остатками', 'conflicting_pct': 'Таких ключей, %',
    'max_values_per_article': 'Максимум значений на артикул',
    'model': 'Модель', 'split': 'Период', 'variant': 'Набор признаков',
    'feature': 'Признак', 'group': 'Группа', 'features': 'Признаков',
    'feature_count': 'Число признаков', 'coefficient': 'Коэффициент', 'importance': 'Важность',
    'fit_seconds': 'Время обучения, с', 'MAE': 'MAE, шт.', 'WAPE': 'WAPE, доля',
    'RMSE': 'RMSE, шт.', 'RMSLE': 'RMSLE', 'bias': 'Смещение, шт.',
    'objective': 'Функция потерь', 'trees': 'Деревьев', 'epoch': 'Эпоха',
    'train_log_mse': 'Ошибка обучения в log-шкале',
    'MAE_change_vs_previous': 'Изменение MAE к предыдущему набору',
    'zero_share': 'Доля нулевых продаж', 'mean_quantity': 'Средние продажи',
    'short_history_share': 'Доля пар моложе 28 дней', 'segment': 'Срез',
    'max_missing': 'Максимальная доля пропусков', 'value': 'Значение',
    'base_valid_MAE': 'MAE на валидации: база', 'fe_valid_MAE': 'MAE на валидации: FE',
    'base_test_MAE': 'MAE на тесте: база', 'fe_test_MAE': 'MAE на тесте: FE',
    'valid_gain_pct': 'Улучшение на валидации, %', 'test_gain_pct': 'Улучшение на тесте, %',
}
for lag in [1, 7, 14, 28]:
    LABELS[f'qty_lag_{lag}'] = f'Продажи {lag} дн. назад'
for window in [7, 28]:
    for stat, title in [('mean','Средние продажи'), ('std','Разброс продаж'), ('max','Максимум продаж')]:
        LABELS[f'qty_roll_{stat}_{window}'] = f'{title} за {window} дн.'
    LABELS[f'qty_nonzero_share_{window}'] = f'Доля дней с продажами за {window} дн.'
for col, title in [('stock_mean','Средний остаток'),('stock_min','Минимальный остаток'),
                   ('stock_max','Максимальный остаток'),('stock_records','Записей остатка')]:
    LABELS[f'{col}_lag_1'] = f'{title} вчера'
for prefix, title in [('warehouse','Склад'),('article','Артикул')]:
    for lag in [1,7]:
        LABELS[f'{prefix}_qty_lag_{lag}'] = f'{title}: продажи {lag} дн. назад'
    LABELS[f'{prefix}_qty_roll_mean_7'] = f'{title}: средние продажи за 7 дн.'


def identifier_label(value):
    """Сократить анонимные идентификаторы только для показа."""
    if not isinstance(value, str):
        return value
    result = value
    for stem, title in [('char_article','Артикул'),('warehouse_name','Склад'),('cities_name','Город'),
                        ('composite_division','Подразделение'),('category_lvl_1','Категория 1'),
                        ('category_lvl_2','Категория 2'),('category_lvl_3','Категория 3'),
                        ('characteristic','Характеристика')]:
        result = re.sub(rf'{stem}_(\d+)', rf'{title} \1', result)
    return {'MISSING':'Пропуск','OTHER':'Остальные','__MISSING__':'Пропуск',
            '__UNKNOWN_AT_DATE__':'Ещё не известно'}.get(result,result)


def feature_label(value, width=None):
    """Название столбца или коэффициента; технический код остаётся в словаре."""
    if not isinstance(value, str):
        return value
    clean = value.removeprefix('numeric__').removeprefix('category__')
    result = LABELS.get(clean, MODEL_LABELS.get(clean, GROUP_LABELS.get(clean)))
    if result is None:
        if clean.startswith('missingindicator_'):
            result = 'Пропуск: ' + feature_label(clean.removeprefix('missingindicator_'))
        else:
            for column in sorted(LABELS, key=len, reverse=True):
                if clean.startswith(column+'_'):
                    result = LABELS[column]+' = '+identifier_label(clean[len(column)+1:])
                    break
    result = result or identifier_label(clean)
    return textwrap.fill(result,width=width) if width else result


def display_ru(value, **kwargs):
    """Показать копию таблицы с русскими подписями, сохранив исходные данные."""
    if isinstance(value,pd.Series):
        value=value.to_frame()
    if isinstance(value,pd.DataFrame):
        value=value.copy()
        for c in value.select_dtypes(include=['object','string','category']).columns:
            value[c]=value[c].astype(object).map(lambda x: feature_label(x) if isinstance(x,str) else x)
        value=value.rename(index=feature_label,columns=feature_label)
        value.index.name=feature_label(value.index.name)
        value.columns.name=feature_label(value.columns.name)
    display(value,**kwargs)
