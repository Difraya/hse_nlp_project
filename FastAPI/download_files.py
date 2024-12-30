import aiohttp
import asyncio
import time

successful_downloads = 0

async def download_file_from_yandex_disk(url, local_filename):
    global successful_downloads

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                with open(local_filename, 'wb') as f:
                    content = await response.read()
                    f.write(content)
                successful_downloads += 1
                print(f"Файл {successful_downloads}-{local_filename} успешно загружен.")
        except aiohttp.ClientResponseError as e:
            print(f"Не удалось загрузить {local_filename}: {e.status}, {e.message}")
        except Exception as e:
            print(f"Произошла ошибка: {e}")


# Ссылки для скачивания больших файлов с Яндекс-Диска
files = {
    'model1': 'https://downloader.disk.yandex.ru/disk/6d21c4c26be24f666b2faa2257bbe12f7fd9eb828f26723c6e66cfe6890b896f/6772ba41/Mfih3zJy9UfpjGfa2ErvjyPV5rJvZ2aKnTcow0NtFuyQ_dkCsNLoxIiq1gcLoyZCP60t8Q-tt6Kno8QujrUPIw%3D%3D?uid=0&filename=pipeline.joblib&disposition=attachment&hash=UaA%2B%2BN8Kfg4PiT1YHOcIq/uE7aFEOIk/lYFY18vbgpqequyDI35/G2N8HWzvw3HLq/J6bpmRyOJonT3VoXnDag%3D%3D%3A&limit=0&content_type=application%2Foctet-stream&owner_uid=1008179610&fsize=2281419899&hid=e47d3089b842439879c17c704af7ea5b&media_type=data&tknv=v2',
    'model2': 'https://downloader.disk.yandex.ru/disk/11ae401ded27117efa8097c4d76f2e53a13dbe6127f02763ac5cea3600710ef4/6772bacd/Mfih3zJy9UfpjGfa2Ervj00w1YWbyftzoHqIKx9YwX1HH0S9kOrzip7BkPmsTg-Nzru1AogxuKCK2bNpsoccIw%3D%3D?uid=0&filename=ngram_naive_bayes.joblib&disposition=attachment&hash=A%2Bw%2BuAQxeKpsz1RwDa5/yUCPZ1ecPFkCq/Oq56WWEJj6zVlzXXZmoBnUsu/J%2B/iqq/J6bpmRyOJonT3VoXnDag%3D%3D%3A&limit=0&content_type=application%2Foctet-stream&owner_uid=1008179610&fsize=825770&hid=26b08a48d6e5f525e2cb678f952b7b74&media_type=data&tknv=v2',
    'model3': 'https://downloader.disk.yandex.ru/disk/4a29a1fbf33056527ddc305a179fc4ae11c6addebe5db4a7af88c88ad3577e36/6772baef/Mfih3zJy9UfpjGfa2Ervjz6ywREt8tF_cF7mpX4BDTeAz-3M2xC-OZjvJLAQK9XRjRk-pesUou219TKy6opx6Q%3D%3D?uid=0&filename=tfidf_log_reg_standardized.joblib&disposition=attachment&hash=2XANsUhI7TPNlbH7spQgi5WjeQ2N94SEXPEZONvEgkkJiPZnh4YRhagcFyzNv3rYq/J6bpmRyOJonT3VoXnDag%3D%3D%3A&limit=0&content_type=application%2Foctet-stream&owner_uid=1008179610&fsize=289545&hid=30219a8f2a1746736e9607effc3d83ee&media_type=data&tknv=v2',
    'model4': 'https://downloader.disk.yandex.ru/disk/78fb75fe8fafdb5d0b62d7ee8d7e90146b823d9450fffc173926110572ac00a5/6772bb10/Mfih3zJy9UfpjGfa2Ervj1tTY7oOk12EoZN5xEkOZelZ8YUiqLOINYLnJUrIPC8aWiWgLVaFvKakpPI0qseywQ%3D%3D?uid=0&filename=SGDClassifier.joblib&disposition=attachment&hash=ItnkxwBCSsxXo8yNTmJYU/p5mfQ86mNKS2y9r9Sl/zFUf/k7UarSjM7foBr/7p8lq/J6bpmRyOJonT3VoXnDag%3D%3D%3A&limit=0&content_type=application%2Foctet-stream&owner_uid=1008179610&fsize=613391&hid=bf06631a00b1b42cb779fc4e1f38c4dc&media_type=data&tknv=v2',
    'df_train': 'https://downloader.disk.yandex.ru/disk/04a323823a4efdb01c59708f485fe52687ecf029c6bbb7335b2acf25c90b1993/6772bb36/Mfih3zJy9UfpjGfa2Ervj1x6opftuSvJ_ioVWY8yCszRBd87WsdangJrmntG9Y_N5cvYOODhbXaOrYlFAY7Y9A%3D%3D?uid=0&filename=df_train.pq&disposition=attachment&hash=VCZmSIGG5YSmqAr4jWmXKEjLFfHzYlAEANPjQR8KzLIyGPYIXMlW%2Bz0W%2BT7faWgtq/J6bpmRyOJonT3VoXnDag%3D%3D%3A&limit=0&content_type=application%2Foctet-stream&owner_uid=1008179610&fsize=145583295&hid=53fff03d1562b538221956cc14fdd47b&media_type=data&tknv=v2',
    'df_test': 'https://downloader.disk.yandex.ru/disk/b3e4c5e792d539a94b8ff06871d9fec0ddeadbb16d64881a1a855b10e8c1bf8a/6772bb4e/Mfih3zJy9UfpjGfa2Ervj6iXANtQy_KIcOjtht_beum8neGe5hJ2aZ4AsBmbnbfJ_3_b1Ay2tSgMuy39AwZEmw%3D%3D?uid=0&filename=df_test.pq&disposition=attachment&hash=EfH915hD3wYvvu3CP2h05UG5eXIc2n8vixVauzLhu/ht5M5bkKs3JELsGVjGYqx/q/J6bpmRyOJonT3VoXnDag%3D%3D%3A&limit=0&content_type=application%2Foctet-stream&owner_uid=1008179610&fsize=75295229&hid=e8e4549005ed04e2f3682859009cb2c9&media_type=data&tknv=v2',
}

# Путь для сохранения загруженных файлов
local_paths = {
    'model1': 'pipeline.joblib',
    'model2': 'ngram_naive_bayes.joblib',
    'model3': 'tfidf_log_reg_standardized.joblib',
    'model4': 'SGDClassifier.joblib',
    'df_train': 'df_train.pq',
    'df_test': 'df_test.pq',
}


# Основная функция для запуска загрузки всех файлов
async def main():
    tasks = []
    print(f"Начинаем скачивание и сохранение {len(files)} файла(ов).")
    start_time = time.time()
    for key, url in files.items():
        tasks.append(download_file_from_yandex_disk(url, local_paths[key]))

    await asyncio.gather(*tasks)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Время выполнения: {elapsed_time:.2f} секунд(ы).")
    print(f"{successful_downloads} файла(ов) успешно загружены.")


# Запуск асинхронных задач
if __name__ == '__main__':
    asyncio.run(main())
