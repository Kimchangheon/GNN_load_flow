import time
from case_generator import case_generation
from newton_raphson import newtonrapson
from database import append_to_parquet, get_parquet_file_size, read_first_row_parquet
import pandas as pd
import numpy as np
import multiprocessing
import threading
import os
import torch
debugging = False
pic = False
U_base = 110e3
S_base = 100 * 1e6
U_base = 1
S_base = 1
runs = 65536
runs = 1

filename = f'u_start_repaired_{str(runs)}_variations_4_8_16_32_bus_grid_Ybus.parquet'
save_steps = 16
fixed = True  # Für Changhun
done = False



def process_run(run_idx, debugging, pic, U_base, S_base, fixed, queue):
    try:

        exponents = np.arange(2, 6)
        bus_number = 2 ** np.random.choice(exponents)
        #print(bus_number)
        bus_typ, s_multi, u_start, Y_matrix, is_connected, Y_Lines, Y_C_Lines, Lines_connected = case_generation(
            bus_number, fixed, debugging, pic, U_base, S_base
        )

        # float 64 type
        Yr, Yi = Y_matrix.real.copy().flatten(), Y_matrix.imag.copy().flatten()

        if not is_connected:
            #print("Alarm")
            df = pd.DataFrame({
                'bus_number': [bus_number],
                'bus_typ': [bus_typ],
                'U_base': [U_base],
                'S_base': [S_base],

                'Y_Lines': [Y_Lines],
                'Y_C_Lines': [Y_C_Lines],
                'Lines_connected': [Lines_connected],
                'Yr': [Yr],
                'Yi': [Yi],

                'u_start': [u_start],
                'u_newton': [np.zeros_like(u_start, dtype=complex)],

                'S_start': [s_multi],
                'S_newton': [np.zeros_like(u_start, dtype=complex)],

                'I_newton': [np.zeros_like(u_start, dtype=complex)],
            })
            queue.put(df)
            return df

        u_newton, I_newton, S_newton = newtonrapson(bus_typ, Y_matrix.copy(), s_multi.copy(), u_start.copy())


        df = pd.DataFrame({
            'bus_number': [bus_number],
            'bus_typ': [bus_typ],
            'U_base': [U_base],
            'S_base': [S_base],

            'Y_Lines': [Y_Lines],
            'Y_C_Lines': [Y_C_Lines],
            'Lines_connected': [Lines_connected],
            'Yr': [Yr],
            'Yi': [Yi],

            'u_start': [u_start],
            'u_newton': [u_newton],

            'S_start': [s_multi],
            'S_newton': [S_newton],

            'I_newton': [I_newton]
        })

        queue.put(df)
    except Exception as e:
        print(f"Fehler in Run {run_idx}: {str(e)}")

def listener(queue, filename, save_steps):
    buffer = []
    count = 0
    while True:
        msg = queue.get()
        #print(f"Received msg of type {type(msg)}: {msg}")
        if isinstance(msg, str) and msg == 'DONE':
            break


        buffer.append(msg)
        count += 1

        if len(buffer) >= save_steps:
            combined = pd.concat(buffer, ignore_index=True)
            append_to_parquet(combined, filename)
            buffer.clear()
            print(f"{count} Runs verarbeitet und gespeichert.")

    # Restliche Daten speichern
    if buffer:
        combined = pd.concat(buffer, ignore_index=True)
        append_to_parquet(combined, filename)
        print(f"Letzte {len(buffer)} Ergebnisse gespeichert.")

def run_parallel():
    startzeit = time.time()
    manager = multiprocessing.Manager()
    queue = manager.Queue()

    listener_process = multiprocessing.Process(target=listener, args=(queue, filename, save_steps))
    listener_process.start()

    num_cpu = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cpu)

    for run_idx in range(runs):
        pool.apply_async(process_run, args=(run_idx, debugging, pic, U_base, S_base, fixed, queue))

    pool.close()
    pool.join()

    queue.put('DONE')
    listener_process.join()

    get_parquet_file_size(filename)
    read_first_row_parquet(filename, 0)
    print(f"Gesamtzeit: {time.time() - startzeit:.2f} Sekunden")

if __name__ == '__main__':
    run_parallel()
