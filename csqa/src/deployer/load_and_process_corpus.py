import logging
import queue
import threading
import click
import os
from pathlib import Path
from utilities.corpus_preprocessing.load_dialogues import get_files_from_direc, load_dialogues_from_json_file

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Global variable needed by threads
queue_lock = None
work_queue = None


def create_dialogue_data_dict(files, queue, thread_name):
    dialogues_dict = dict()
    for file in files:
        log.debug("Process file %s" % file)
        try:
            dialogue_in_file = load_dialogues_from_json_file(file)
        except:
            log.info("Problem with file %s " % (file))
            continue

        parts_of_path = Path(file).parts
        key = os.path.join(parts_of_path[-3], parts_of_path[-2], parts_of_path[-1])
        dialogues_dict[key] = dialogue_in_file
        log.debug("File %s loaded by thread %s" % (file,thread_name))
    queue_lock.acquire()
    queue.put(item=dialogues_dict)
    queue_lock.release()


def splite_list_in_chunks(input_list, num_chunks):
    return [input_list[i::num_chunks] for i in range(num_chunks)]


@click.command()
@click.option('-input', help='path to corpus directory', required=True)
@click.option('-num_threads', help='Define number of threads for processing the data', required=True)
def main(input, num_threads):
    input_direc = input
    num_threads = int(num_threads)
    log.info("Load files from %s" % (input_direc))
    files = get_files_from_direc(input_direc)
    log.info("Files loaded \n")

    # In train directory there are n subdirectories QA_0,...,QA_n
    # Either load all files from all directories or from specific directory. Use -input param
    # For test prupose only specific subdirectory is used
    # Filenames are the keys and the instances in the files are saved as as a list
    dialogues_data_dict = dict()
    chunks_of_files = splite_list_in_chunks(input_list=files, num_chunks=num_threads)
    global queue_lock
    queue_lock = threading.Lock()
    global work_queue
    work_queue = queue.Queue(len(chunks_of_files))
    threads = []

    log.info("Extract dialogues from files")
    for id, current_file_chunk in enumerate(chunks_of_files):
        thread_name = "Thread-" + str(id)
        thread = DataLoaderThread(threadID=id, name=thread_name, files=current_file_chunk, queue=work_queue)
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for t in threads:
        t.join()

    log.info("Dialogues extracted")

    # Get results from queue
    log.info("Collect results computed by threads")
    queue_lock.acquire()
    while not work_queue.empty():
        dialogue_chunk_dict = work_queue.get()
        dialogues_data_dict.update(dialogue_chunk_dict)
    queue_lock.release()

    for key, dialogue in dialogues_data_dict.items():
        print("User: %s \n " % (dialogue[0]))
        print("System %s" % (dialogue[1]))
        print("------------------------\n \n")


class DataLoaderThread(threading.Thread):
    def __init__(self, threadID, name, files, queue):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.files = files
        self.queue = queue

    def run(self):
        log.info("Starting %s" % self.name)
        create_dialogue_data_dict(files=self.files, queue=self.queue, thread_name=self.name)
        log.info("Exiting %s" % self.name)


if __name__ == '__main__':
    main()
