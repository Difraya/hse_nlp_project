## Разведочный анализ данных (EDA) для задачи определения авторства текста
### Введение
В этом проекте наша цель — определить автора данного отрывка текста из пула 100 возможных авторов.
Целью этого разведочного анализа данных является понимание характеристик набора данных, выявление закономерностей, 
обнаружение потенциальных проблем и получение инсайтов, которые будут направлять процесс моделирования.
Нами были выполнены [четыре](https://github.com/Difraya/hse_nlp_project/tree/main/EDA) независимых разведочных анализа на немного различающихся датафреймах, полученных из нашего [датасета](https://www.kaggle.com/datasets/vorvit/books-eng).
Далее приводятся численные значения, полученные при анализе данных в файле [Voronik_EDA.ipynb](https://github.com/Difraya/hse_nlp_project/blob/main/EDA/Voronik_EDA.ipynb).

### Обзор набора данных
Набор данных состоит из коллекции текстов книг, помеченных метками авторов. Основные статистические данные о наборе данных:

- **Общее количество текстов**: 101
- **Количество уникальных авторов**: 101
- **Среднее количество текстов на одного автора**: 1

Распределение длины текста - **Медианная длина текста**: 87383 слова
- **Интерквартильный размах (IQR)**: 39149 - 93173 слов
Длина текста может влиять на производительность модели, по этой причине мы ограничили длину текстов в 500000 символов,
вырезаяя лишние символы из середины, чтобы сохранить, теоретически, наиболее значимые начало и конец книги.

Анализ словарного запаса- **Общий размер словарного запаса**: 149146 уникальных токенов
- **Средний словарный запас на автора**: 8153 уникальных токенов
- **Наиболее распространенные слова**: "one", "said", "would", "like", "could".
- **Наиболее распространенные биграммы**: "old man", "could see", "young man", "let us", "long time".
- **Наиболее распространенные триграммы**: "let us go", "horned mother deer", "gargantua pantagruel francois",
"pantagruel francois rabelais", "herr von knobelsdorff"

Авторский лексический запас - **Лексическое разнообразие**: Каждому автору своейственно своё отношение уникальных слов к общему количеству слов в текстах.
- **Терминология, специфичная для области**: Некоторые авторы используют специализированный лексикон в зависимости от своей области
экспертизы (например, философия, наука, поэзия).

Знаки препинания и стилистические особенности- **Использование знаков препинания**: Значительные различия среди авторов: одни
предпочитают минимальное количество препинаний, другие используют более сложные структуры предложений.
- **Стилистические особенности**: Отличительные стилистические характеристики включают среднюю длину предложения,
частоту употребления имён собственных, различных соварных оборотов.

Распределение классов - **Равномерность**: Количество текстов на автора примерно сбалансировано,
отличается только объем текста, который был нами искусственно ограничен макимальным количеством в 500000 символов,
что снижает опасения по поводу дисбаланса классов при обучении модели.

### Заключение
- Разведочный анализ данных предоставляет полезные инсайты о наборе данных, подчеркивая характеристики текста,
которые могут помочь в различении авторов. Варьирование длины текста, обилие словарного запаса, ключевые слова и n-граммы и
стилистические особенности среди авторов будут ключевыми для обучения эффективной модели классификации.
- Распределение частей речи также указывает на различия в стилях.
- Наши следующие шаги включают инженеринг признаков на основе этих наблюдений и выбор подходящих алгоритмов для многоклассовой классификации.
- Html теги и спецсимволы необходимо дополнительно обработать.
- Также в текстах встречаются слова не только на английском языке.
- Несовременные авторы часто используют нераспространенные в современном языке слова типа thy.
- У многих авторов популярными биграммами являются имена их персонажей, что должно значительно упрощать классификацию.
- Возможно, потребуется разбиение на части длинных текстов для эффективной работы будущей модели классификации, а также замена коротких версий текстов на более длинные для тех же авторов, либо замена автороа на другого.
