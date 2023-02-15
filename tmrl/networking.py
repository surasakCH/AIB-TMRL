# standard library imports
import datetime
import os
import pickle
import select
import socket
import time
import atexit
# import gc
import json
import shutil
import tempfile
from os.path import exists
from copy import deepcopy
from threading import Lock, Thread

# third-party imports
import numpy as np
import torch
from requests import get

# local imports
from tmrl.actor import ActorModule
from tmrl.util import collate, dump, load, partial_to_dict
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj

import logging


__docformat__ = "google"


# PRINT: ============================================


def print_with_timestamp(s):
    x = datetime.datetime.now()
    sx = x.strftime("%x %X ")
    logging.info(sx + str(s))


# NETWORK: ==========================================


def send_ack(sock):
    return send_object(sock, None, ping=False, pong=False, ack=True)


def send_object(sock, obj, ping=False, pong=False, ack=False):
    """
    If ping, this will ignore obj and send the PING request
    If pong, this will ignore obj and send the PONG request
    If ack, this will ignore obj and send the ACK request
    If raw, obj must be a binary string
    Call only after select on a socket with a (long enough) timeout.
    Returns True if sent successfully, False if connection lost.
    """
    if ping:
        msg = bytes(f"{'PING':<{cfg.HEADER_SIZE}}", 'utf-8')
    elif pong:
        msg = bytes(f"{'PONG':<{cfg.HEADER_SIZE}}", 'utf-8')
    elif ack:
        msg = bytes(f"{'ACK':<{cfg.HEADER_SIZE}}", 'utf-8')
    else:
        msg = pickle.dumps(obj)
        msg = bytes(f"{len(msg):<{cfg.HEADER_SIZE}}", 'utf-8') + msg
        if cfg.PRINT_BYTESIZES:
            print_with_timestamp(f"Sending {len(msg)} bytes.")
    try:
        sock.sendall(msg)
    except OSError:  # connection closed or broken
        return False
    return True


def recv_object(sock):
    """
    If the request is PING or PONG, this will return 'PINGPONG'
    If the request is ACK, this will return 'ACK'
    If the request is PING, this will automatically send the PONG answer
    Call only after select on a socket with a (long enough) timeout.
    Returns the object if received successfully, None if connection lost.
    This sends the ACK request back to sock when an object transfer is complete
    """
    # first, we receive the header (inefficient but prevents collisions)
    msg = b''
    l = len(msg)
    while l != cfg.HEADER_SIZE:
        try:
            recv_msg = sock.recv(cfg.HEADER_SIZE - l)
            if len(recv_msg) == 0:  # connection closed or broken
                return None
            msg += recv_msg
        except OSError:  # connection closed or broken
            return None
        l = len(msg)
    if msg[:3] == b'ACK':
        return 'ACK'
    msglen = int(msg[:cfg.HEADER_SIZE])
    # now, we receive the actual data (no more than the data length, again to prevent collisions)
    msg = b''
    l = len(msg)
    while l != msglen:
        try:
            recv_msg = sock.recv(min(cfg.BUFFER_SIZE, msglen - l))  # this will not receive more bytes than required
            if len(recv_msg) == 0:  # connection closed or broken
                return None
            msg += recv_msg
        except OSError:  # connection closed or broken
            return None
        l = len(msg)
    send_ack(sock)
    return pickle.loads(msg)


def get_listening_socket(timeout, ip_bind, port_bind):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # to reuse address on Linux
    s.bind((ip_bind, port_bind))
    s.listen(5)
    return s


def get_connected_socket(timeout, ip_connect, port_connect):
    """
    returns the connected socket
    returns None if connect failed
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((ip_connect, port_connect))
    except OSError:  # connection broken or timeout
        print_with_timestamp(f"connect() timed-out or failed, sleeping {cfg.WAIT_BEFORE_RECONNECTION}s")
        s.close()
        time.sleep(cfg.WAIT_BEFORE_RECONNECTION)
        return None
    s.settimeout(cfg.SOCKET_TIMEOUT_COMMUNICATE)
    return s


def accept_or_close_socket(s):
    """
    returns conn, addr
    None None in case of failure
    """
    conn = None
    try:
        conn, addr = s.accept()
        conn.settimeout(cfg.SOCKET_TIMEOUT_COMMUNICATE)
        return conn, addr
    except OSError:
        if conn is not None:
            conn.close()
        s.close()
        time.sleep(cfg.WAIT_BEFORE_RECONNECTION)
        return None, None


def select_and_send_or_close_socket(obj, conn):
    """
    Returns True if success
    False if disconnected (closes sockets)
    """
    print_with_timestamp(f"start select")
    _, wl, xl = select.select([], [conn], [conn], cfg.SELECT_TIMEOUT_OUTBOUND)  # select for writing
    print_with_timestamp(f"end select")
    if len(xl) != 0:
        print_with_timestamp("error when writing, closing socket")
        conn.close()
        return False
    if len(wl) == 0:
        print_with_timestamp("outbound select() timed out, closing socket")
        conn.close()
        return False
    elif not send_object(conn, obj):  # error or timeout
        print_with_timestamp("send_object() failed, closing socket")
        conn.close()
        return False
    return True


def poll_and_recv_or_close_socket(conn):
    """
    Returns True, obj is success (obj is None if nothing was in the read buffer when polling)
    False, None otherwise
    """
    rl, _, xl = select.select([conn], [], [conn], 0.0)  # polling read channel
    if len(xl) != 0:
        print_with_timestamp("error when polling, closing sockets")
        conn.close()
        return False, None
    if len(rl) == 0:  # nothing in the recv buffer
        return True, None
    obj = recv_object(conn)
    if obj is None:  # socket error
        print_with_timestamp("error when receiving object, closing sockets")
        conn.close()
        return False, None
    elif obj == 'PINGPONG':
        return True, None
    else:
        return True, obj


# BUFFER: ===========================================


class Buffer:
    """
    Buffer of training samples.

    `Server`, `RolloutWorker` and `Trainer` all have their own `Buffer` to store and send training samples.

    Samples are tuples of the form (`act`, `new_obs`, `rew`, `terminated`, `truncated`, `info`)
    """
    def __init__(self, maxlen=cfg.BUFFERS_MAXLEN):
        """
        Args:
            maxlen (int): buffer length
        """
        self.memory = []
        self.stat_train_return = 0.0  # stores the train return
        self.stat_test_return = 0.0  # stores the test return
        self.stat_train_steps = 0  # stores the number of steps per training episode
        self.stat_test_steps = 0  # stores the number of steps per test episode
        self.maxlen = maxlen

    def clip_to_maxlen(self):
        lenmem = len(self.memory)
        if lenmem > self.maxlen:
            print_with_timestamp("buffer overflow. Discarding old samples.")
            self.memory = self.memory[(lenmem - self.maxlen):]

    def append_sample(self, sample):
        """
        Appends `sample` to the buffer.

        Args:
            sample (Tuple): a training sample of the form (`act`, `new_obs`, `rew`, `terminated`, `truncated`, `info`)
        """
        self.memory.append(sample)
        self.clip_to_maxlen()

    def clear(self):
        """
        Clears the buffer but keeps train and test returns.
        """
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def __iadd__(self, other):
        self.memory += other.memory
        self.clip_to_maxlen()
        self.stat_train_return = other.stat_train_return
        self.stat_test_return = other.stat_test_return
        self.stat_train_steps = other.stat_train_steps
        self.stat_test_steps = other.stat_test_steps
        return self


# SERVER SERVER: =====================================


class Server:
    """
    Central server.

    The `Server` lets 1 `Trainer` and n `RolloutWorkers` connect.
    It buffers experiences sent by workers and periodically sends these to the trainer.
    It also receives the weights from the trainer and broadcasts these to the connected workers.
    """
    def __init__(self, min_samples_per_server_packet=1):
        """
        Args:
            min_samples_per_server_packet (int): Minimum number of samples that the
                server buffers from connected workers before sending to the trainer.
        """
        self.__buffer = Buffer()
        self.__buffer_lock = Lock()
        self.__weights_lock = Lock()
        self.__weights = None
        self.__weights_id = 0  # this increments each time new weights are received
        self.samples_per_server_batch = min_samples_per_server_packet
        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())

        print_with_timestamp(f"INFO SERVER: local IP: {self.local_ip}")
        print_with_timestamp(f"INFO SERVER: public IP: {self.public_ip}")

        Thread(target=self.__rollout_workers_thread, args=('', ), kwargs={}, daemon=True).start()
        Thread(target=self.__trainers_thread, args=('', ), kwargs={}, daemon=True).start()

    def __trainers_thread(self, ip):
        """
        This waits for new potential Trainers to connect
        When a new Trainer connects, this instantiates a new thread to handle it
        """
        while True:  # main server loop
            s = get_listening_socket(cfg.SOCKET_TIMEOUT_ACCEPT_TRAINER, ip, cfg.PORT_TRAINER)
            conn, addr = accept_or_close_socket(s)
            if conn is None:
                continue
            print_with_timestamp(f"INFO TRAINERS THREAD: server connected by trainer at address {addr}")
            Thread(target=self.__trainer_thread, args=(conn, ), kwargs={}, daemon=True).start()  # we don't keep track of this for now
            s.close()

    def __trainer_thread(self, conn):
        """
        This periodically sends the local buffer to the TrainerInterface (when data is available)
        When the TrainerInterface sends new weights, this broadcasts them to all connected RolloutWorkers
        """
        ack_time = time.time()
        wait_ack = False
        while True:
            # send samples
            self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
            if len(self.__buffer) >= self.samples_per_server_batch:
                if not wait_ack:
                    obj = self.__buffer
                    if select_and_send_or_close_socket(obj, conn):
                        wait_ack = True
                        ack_time = time.time()
                    else:
                        print_with_timestamp("failed sending object to trainer")
                        self.__buffer_lock.release()
                        break
                    self.__buffer.clear()
                else:
                    elapsed = time.time() - ack_time
                    print_with_timestamp(f"CAUTION: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                    if elapsed >= cfg.ACK_TIMEOUT_SERVER_TO_TRAINER:
                        print_with_timestamp("ACK timed-out, breaking connection")
                        conn.close()
                        self.__buffer_lock.release()
                        wait_ack = False
                        break
            self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
            # checks for weights
            success, obj = poll_and_recv_or_close_socket(conn)
            if not success:
                print_with_timestamp("poll failed in trainer thread")
                break
            elif obj is not None and obj != 'ACK':
                print_with_timestamp(f"trainer thread received obj")
                self.__weights_lock.acquire()  # WEIGHTS LOCK.......................................................
                self.__weights = obj
                self.__weights_id += 1
                self.__weights_lock.release()  # END WEIGHTS LOCK...................................................
            elif obj == 'ACK':
                wait_ack = False
                print_with_timestamp(f"transfer acknowledgment received after {time.time() - ack_time}s")
            time.sleep(cfg.LOOP_SLEEP_TIME)  # TODO: adapt

    def __rollout_workers_thread(self, ip):
        """
        This waits for new potential RolloutWorkers to connect
        When a new RolloutWorker connects, this instantiates a new thread to handle it
        """
        while True:  # main server loop
            s = get_listening_socket(cfg.SOCKET_TIMEOUT_ACCEPT_ROLLOUT, ip, cfg.PORT_ROLLOUT)
            conn, addr = accept_or_close_socket(s)
            if conn is None:
                continue
            print_with_timestamp(f"INFO WORKERS THREAD: server connected by worker at address {addr}")
            Thread(target=self.__rollout_worker_thread, args=(conn, ), kwargs={}, daemon=True).start()  # we don't keep track of this for now
            s.close()

    def __rollout_worker_thread(self, conn):
        """
        Thread handling connection to a single RolloutWorker
        """
        # last_ping = time.time()
        worker_weights_id = 0
        ack_time = time.time()
        wait_ack = False
        while True:
            # send weights
            self.__weights_lock.acquire()  # WEIGHTS LOCK...............................................................
            if worker_weights_id != self.__weights_id:  # new weigths
                if not wait_ack:
                    obj = self.__weights
                    if select_and_send_or_close_socket(obj, conn):
                        ack_time = time.time()
                        wait_ack = True
                    else:
                        self.__weights_lock.release()
                        print_with_timestamp("select_and_send_or_close_socket failed in worker thread")
                        break
                    worker_weights_id = self.__weights_id
                else:
                    elapsed = time.time() - ack_time
                    print_with_timestamp(f"object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                    if elapsed >= cfg.ACK_TIMEOUT_SERVER_TO_WORKER:
                        print_with_timestamp("ACK timed-out, breaking connection")
                        conn.close()
                        self.__weights_lock.release()
                        # wait_ack = False  # not needed since we end the thread
                        break
            self.__weights_lock.release()  # END WEIGHTS LOCK...........................................................
            # checks for samples
            success, obj = poll_and_recv_or_close_socket(conn)
            if not success:
                print_with_timestamp("poll failed in rollout thread")
                break
            elif obj is not None and obj != 'ACK':
                print_with_timestamp(f"rollout worker thread received obj")
                self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
                self.__buffer += obj  # concat worker batch to local batch
                self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
            elif obj == 'ACK':
                wait_ack = False
                print_with_timestamp(f"transfer acknowledgment received after {time.time() - ack_time}s")
            time.sleep(cfg.LOOP_SLEEP_TIME)  # TODO: adapt


# TRAINER: ==========================================


class TrainerInterface:
    """
    This is the trainer's network interface
    This connects to the server
    This receives samples batches and sends new weights
    """
    def __init__(self, server_ip=None, model_path=cfg.MODEL_PATH_TRAINER):
        self.__buffer_lock = Lock()
        self.__weights_lock = Lock()
        self.__weights = None
        self.__buffer = Buffer()
        self.model_path = model_path
        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'
        self.recv_tiemout = cfg.RECV_TIMEOUT_TRAINER_FROM_SERVER

        print_with_timestamp(f"local IP: {self.local_ip}")
        print_with_timestamp(f"public IP: {self.public_ip}")
        print_with_timestamp(f"server IP: {self.server_ip}")

        Thread(target=self.__run_thread, args=(), kwargs={}, daemon=True).start()

    def __run_thread(self):
        """
        Trainer interface thread
        """
        while True:  # main client loop
            ack_time = time.time()
            recv_time = time.time()
            wait_ack = False
            s = get_connected_socket(cfg.SOCKET_TIMEOUT_CONNECT_TRAINER, self.server_ip, cfg.PORT_TRAINER)
            if s is None:
                print_with_timestamp("get_connected_socket failed in TrainerInterface thread")
                continue
            while True:
                # send weights
                self.__weights_lock.acquire()  # WEIGHTS LOCK...........................................................
                if self.__weights is not None:  # new weights
                    if not wait_ack:
                        obj = self.__weights
                        if select_and_send_or_close_socket(obj, s):
                            ack_time = time.time()
                            wait_ack = True
                        else:
                            self.__weights_lock.release()
                            print_with_timestamp("select_and_send_or_close_socket failed in TrainerInterface")
                            break
                        self.__weights = None
                    else:
                        elapsed = time.time() - ack_time
                        print_with_timestamp(f"CAUTION: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= cfg.ACK_TIMEOUT_TRAINER_TO_SERVER:
                            print_with_timestamp("ACK timed-out, breaking connection")
                            self.__weights_lock.release()
                            wait_ack = False
                            break
                self.__weights_lock.release()  # END WEIGHTS LOCK.......................................................
                # checks for samples batch
                success, obj = poll_and_recv_or_close_socket(s)
                if not success:
                    print_with_timestamp("poll failed in TrainerInterface thread")
                    break
                elif obj is not None and obj != 'ACK':  # received buffer
                    print_with_timestamp(f"trainer interface received obj")
                    recv_time = time.time()
                    self.__buffer_lock.acquire()  # BUFFER LOCK.........................................................
                    self.__buffer += obj
                    self.__buffer_lock.release()  # END BUFFER LOCK.....................................................
                elif obj == 'ACK':
                    wait_ack = False
                    print_with_timestamp(f"transfer acknowledgment received after {time.time() - ack_time}s")
                elif time.time() - recv_time > self.recv_tiemout:
                    print_with_timestamp(f"Timeout in TrainerInterface, not received anything for too long")
                    break
                time.sleep(cfg.LOOP_SLEEP_TIME)  # TODO: adapt
            s.close()

    def broadcast_model(self, model: ActorModule):
        """
        model must be an ActorModule
        broadcasts the model's weights to all connected RolloutWorkers
        """
        t0 = time.time()
        self.__weights_lock.acquire()  # WEIGHTS LOCK...................................................................
        t1 = time.time()
        torch.save(model.state_dict(), self.model_path)
        t2 = time.time()
        with open(self.model_path, 'rb') as f:
            self.__weights = f.read()
        t3 = time.time()
        self.__weights_lock.release()  # END WEIGHTS LOCK...............................................................
        print_with_timestamp(f"broadcast_model: lock acquire: {t1 - t0}s, save dict: {t2 - t1}s, read dict: {t3 - t2}s")

    def retrieve_buffer(self):
        """
        returns a copy of the TrainerInterface's local buffer, and clears it
        """
        self.__buffer_lock.acquire()  # BUFFER LOCK.....................................................................
        buffer_copy = deepcopy(self.__buffer)
        self.__buffer.clear()
        self.__buffer_lock.release()  # END BUFFER LOCK.................................................................
        return buffer_copy


def log_environment_variables():
    """
    add certain relevant environment variables to our config
    usage: `LOG_VARIABLES='HOME JOBID' python ...`
    """
    return {k: os.environ.get(k, '') for k in os.environ.get('LOG_VARIABLES', '').strip().split()}


def load_run_instance(checkpoint_path):
    """
    Default function used to load trainers from checkpoint path
    Args:
        checkpoint_path: the path where instances of run_cls are checkpointed
    Returns:
        An instance of run_cls loaded from checkpoint_path
    """
    return load(checkpoint_path)


def dump_run_instance(run_instance, checkpoint_path):
    """
    Default function used to dump trainers to checkpoint path
    Args:
        run_instance: the instance of run_cls to checkpoint
        checkpoint_path: the path where instances of run_cls are checkpointed
    """
    dump(run_instance, checkpoint_path)


def iterate_epochs_tm(run_cls,
                      interface: TrainerInterface,
                      checkpoint_path: str,
                      dump_run_instance_fn=dump_run_instance,
                      load_run_instance_fn=load_run_instance,
                      epochs_between_checkpoints=1,
                      updater_fn=None):
    """
    Main training loop (remote)
    The run_cls instance is saved in checkpoint_path at the end of each epoch
    The model weights are sent to the RolloutWorker every model_checkpoint_interval epochs
    Generator yielding episode statistics (list of pd.Series) while running and checkpointing
    """
    checkpoint_path = checkpoint_path or tempfile.mktemp("_remove_on_exit")

    try:
        logging.debug(f"checkpoint_path: {checkpoint_path}")
        if not exists(checkpoint_path):
            logging.info(f"=== specification ".ljust(70, "="))
            run_instance = run_cls()
            dump_run_instance_fn(run_instance, checkpoint_path)
            logging.info(f"")
        else:
            logging.info(f"Loading checkpoint...")
            t1 = time.time()
            run_instance = load_run_instance_fn(checkpoint_path)
            logging.info(f" Loaded checkpoint in {time.time() - t1} seconds.")
            if updater_fn is not None:
                logging.info(f"Updating checkpoint...")
                t1 = time.time()
                run_instance = updater_fn(run_instance, run_cls)
                logging.info(f"Checkpoint updated in {time.time() - t1} seconds.")

        while run_instance.epoch < run_instance.epochs:
            # time.sleep(1)  # on network file systems writing files is asynchronous and we need to wait for sync
            yield run_instance.run_epoch(interface=interface)  # yield stats data frame (this makes this function a generator)
            if run_instance.epoch % epochs_between_checkpoints == 0:
                logging.info(f" saving checkpoint...")
                t1 = time.time()
                dump_run_instance_fn(run_instance, checkpoint_path)
                logging.info(f" saved checkpoint in {time.time() - t1} seconds.")
                # we delete and reload the run_instance from disk to ensure the exact same code runs regardless of interruptions
                # del run_instance
                # gc.collect()  # garbage collection
                # run_instance = load_run_instance_fn(checkpoint_path)

    finally:
        if checkpoint_path.endswith("_remove_on_exit") and exists(checkpoint_path):
            os.remove(checkpoint_path)


def run_with_wandb(entity, project, run_id, interface, run_cls, checkpoint_path: str = None, dump_run_instance_fn=None, load_run_instance_fn=None, updater_fn=None):
    """
    Main training loop (remote).

    saves config and stats to https://wandb.com
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance
    wandb_dir = tempfile.mkdtemp()  # prevent wandb from polluting the home directory
    atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)  # clean up after wandb atexit handler finishes
    import wandb
    logging.debug(f" run_cls: {run_cls}")
    config = partial_to_dict(run_cls)
    config['environ'] = log_environment_variables()
    # config['git'] = git_info()  # TODO: check this for bugs
    resume = checkpoint_path and exists(checkpoint_path)
    wandb_initialized = False
    err_cpt = 0
    while not wandb_initialized:
        try:
            wandb.init(dir=wandb_dir, entity=entity, project=project, id=run_id, resume=resume, config=config)
            wandb_initialized = True
        except Exception as e:
            err_cpt += 1
            logging.warning(f"wandb error {err_cpt}: {e}")
            if err_cpt > 10:
                logging.warning(f"Could not connect to wandb, aborting.")
                exit()
            else:
                time.sleep(10.0)
    # logging.info(config)
    for stats in iterate_epochs_tm(run_cls, interface, checkpoint_path, dump_run_instance_fn, load_run_instance_fn, 1, updater_fn):
        [wandb.log(json.loads(s.to_json())) for s in stats]


def run(interface, run_cls, checkpoint_path: str = None, dump_run_instance_fn=None, load_run_instance_fn=None, updater_fn=None):
    """
    Main training loop (remote).
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance
    for stats in iterate_epochs_tm(run_cls, interface, checkpoint_path, dump_run_instance_fn, load_run_instance_fn, 1, updater_fn):
        pass


class Trainer:
    """
    Training entity.

    The `Trainer` object is where RL training happens.
    Typically, it can be located on a HPC cluster.
    """
    def __init__(self,
                 training_cls=cfg_obj.TRAINER,
                 server_ip=cfg.SERVER_IP_FOR_TRAINER,
                 model_path=cfg.MODEL_PATH_TRAINER,
                 checkpoint_path=cfg.CHECKPOINT_PATH,
                 dump_run_instance_fn: callable = None,
                 load_run_instance_fn: callable = None,
                 updater_fn: callable = None):
        """
        Args:
            training_cls (type): training class (subclass of tmrl.training_offline.TrainingOffline)
            server_ip (str): ip of the central `Server`
            model_path (str): path where a local copy of the model will be saved
            checkpoint_path: path where the `Trainer` will be checkpointed (`None` = no checkpointing)
            dump_run_instance_fn (callable): custom serializer (`None` = pickle.dump)
            load_run_instance_fn (callable): custom deserializer (`None` = pickle.load)
            updater_fn (callable): custom updater (`None` = no updater). If provided, this must be a function \
            that takes a checkpoint and training_cls as argument and returns an updated checkpoint. \
            The updater is called after a checkpoint is loaded, e.g., to update your checkpoint with new arguments.
        """
        self.checkpoint_path = checkpoint_path
        self.dump_run_instance_fn = dump_run_instance_fn
        self.load_run_instance_fn = load_run_instance_fn
        self.updater_fn = updater_fn
        self.training_cls = training_cls
        self.interface = TrainerInterface(server_ip=server_ip,
                                          model_path=model_path)

    def run(self):
        """
        Runs training.
        """
        run(interface=self.interface,
            run_cls=self.training_cls,
            checkpoint_path=self.checkpoint_path,
            dump_run_instance_fn=self.dump_run_instance_fn,
            load_run_instance_fn=self.load_run_instance_fn,
            updater_fn=self.updater_fn)

    def run_with_wandb(self,
                       entity=cfg.WANDB_ENTITY,
                       project=cfg.WANDB_PROJECT,
                       run_id=cfg.WANDB_RUN_ID,
                       key=None):
        """
        Runs training while logging metrics to wandb_.

        .. _wandb: https://wandb.ai

        Args:
            entity (str): wandb entity
            project (str): wandb project
            run_id (str): name of the run
            key (str): wandb API key
        """
        if key is not None:
            os.environ['WANDB_API_KEY'] = key
        run_with_wandb(entity=entity,
                       project=project,
                       run_id=run_id,
                       interface=self.interface,
                       run_cls=self.training_cls,
                       checkpoint_path=self.checkpoint_path,
                       dump_run_instance_fn=self.dump_run_instance_fn,
                       load_run_instance_fn=self.load_run_instance_fn,
                       updater_fn=self.updater_fn)


# ROLLOUT WORKER: ===================================


class RolloutWorker:
    """Actor.

    A `RolloutWorker` deploys the current policy in the environment.
    A `RolloutWorker` may connect to a `Server` to which it sends buffered experience.
    Alternatively, it may exist in standalone mode for deployment.
    """
    def __init__(
            self,
            env_cls,
            actor_module_cls,
            sample_compressor: callable = None,
            device="cpu",
            server_ip=None,
            min_samples_per_worker_packet=1,
            max_samples_per_episode=np.inf,
            model_path=cfg.MODEL_PATH_WORKER,
            obs_preprocessor: callable = None,
            crc_debug=False,
            model_path_history=cfg.MODEL_PATH_SAVE_HISTORY,
            model_history=cfg.MODEL_HISTORY,
            standalone=False
    ):
        """
        Args:
            env_cls (type): class of the Gym environment (subclass of tmrl.envs.GenericGymEnv)
            actor_module_cls (type): class of the module containing the policy (subclass of tmrl.actor.ActorModule)
            sample_compressor (callable): compressor for sending samples over the Internet
            device (str): device on which the policy is running
            server_ip (str): ip of the central server
            min_samples_per_worker_packet (int): the worker waits for this number of samples before sending
            max_samples_per_episode (int): if an episode gets longer than this, it is reset
            model_path (str): path where a local copy of the policy will be stored
            obs_preprocessor (callable): utility for modifying observations retrieved from the environment
            crc_debug (bool): can be used for debugging the pipeline
            model_path_history (str): (omit .pth) an history of policies can be stored here
            model_history (int): new policies are saved % model_history (0: not saved)
            standalone (bool): If True, the worker will not try to connect to a server
        """
        self.obs_preprocessor = obs_preprocessor
        self.get_local_buffer_sample = sample_compressor
        self.env = env_cls()
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.model_path = model_path
        self.model_path_history = model_path_history
        self.actor = actor_module_cls(observation_space=obs_space, action_space=act_space).to(device)
        self.device = device
        self.standalone = standalone
        if os.path.isfile(self.model_path):
            logging.debug(f"Loading model from {self.model_path}")
            self.actor.load_state_dict(torch.load(self.model_path, map_location=self.device))
        else:
            logging.debug(f"No model found at {self.model_path}")
        self.buffer = Buffer()
        self.__buffer = Buffer()  # deepcopy for sending
        self.__buffer_lock = Lock()
        self.__weights = None
        self.__weights_lock = Lock()
        self.samples_per_worker_batch = min_samples_per_worker_packet
        self.max_samples_per_episode = max_samples_per_episode
        self.crc_debug = crc_debug
        self.model_history = model_history
        self._cur_hist_cpt = 0

        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'
        self.recv_timeout = cfg.RECV_TIMEOUT_WORKER_FROM_SERVER

        print_with_timestamp(f"local IP: {self.local_ip}")
        print_with_timestamp(f"public IP: {self.public_ip}")
        print_with_timestamp(f"server IP: {self.server_ip}")

        if not self.standalone:
            Thread(target=self.__run_thread, args=(), kwargs={}, daemon=True).start()

    def __run_thread(self):
        """
        Redis thread
        """
        while True:  # main client loop
            ack_time = time.time()
            recv_time = time.time()
            wait_ack = False
            s = get_connected_socket(cfg.SOCKET_TIMEOUT_CONNECT_ROLLOUT, self.server_ip, cfg.PORT_ROLLOUT)
            if s is None:
                print_with_timestamp("get_connected_socket failed in worker")
                continue
            while True:
                # send buffer
                self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
                if len(self.__buffer) >= self.samples_per_worker_batch:  # a new batch is available
                    print_with_timestamp("new batch available")
                    if not wait_ack:
                        obj = self.__buffer
                        if select_and_send_or_close_socket(obj, s):
                            ack_time = time.time()
                            wait_ack = True
                        else:
                            self.__buffer_lock.release()
                            print_with_timestamp("select_and_send_or_close_socket failed in worker")
                            break
                        self.__buffer.clear()  # empty sent batch
                    else:
                        elapsed = time.time() - ack_time
                        print_with_timestamp(f"CAUTION: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= cfg.ACK_TIMEOUT_WORKER_TO_SERVER:
                            print_with_timestamp("ACK timed-out, breaking connection")
                            self.__buffer_lock.release()
                            wait_ack = False
                            break
                self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
                # checks for new weights
                success, obj = poll_and_recv_or_close_socket(s)
                if not success:
                    print_with_timestamp(f"rollout worker poll failed")
                    break
                elif obj is not None and obj != 'ACK':
                    print_with_timestamp(f"rollout worker received obj")
                    recv_time = time.time()
                    self.__weights_lock.acquire()  # WEIGHTS LOCK.......................................................
                    self.__weights = obj
                    self.__weights_lock.release()  # END WEIGHTS LOCK...................................................
                elif obj == 'ACK':
                    wait_ack = False
                    print_with_timestamp(f"transfer acknowledgment received after {time.time() - ack_time}s")
                elif time.time() - recv_time > self.recv_timeout:
                    print_with_timestamp(f"Timeout in RolloutWorker, not received anything for too long")
                    break
                time.sleep(cfg.LOOP_SLEEP_TIME)  # TODO: adapt
            s.close()

    def act(self, obs, test=False):
        """
        Converts inputs to torch tensors and converts outputs to numpy arrays.

        Args:
            obs (nested structure): observation
            test (bool): directly passed to the `act()` method of the `ActorModule`

        Returns:
            action (numpy.array): action computed by the `ActorModule`
        """
        # if self.obs_preprocessor is not None:
        #     obs = self.obs_preprocessor(obs)
        obs = collate([obs], device=self.device)
        with torch.no_grad():
            action = self.actor.act(obs, test=test)
        return action

    def reset(self, collect_samples):
        """
        Starts a new episode.

        Args:
            collect_samples (bool): if True, samples are buffered and sent to the `Server`

        Returns:
            obs (nested structure): observation retrieved from the environment
            info (dict): information retrieved from the environment
        """
        obs = None
        act = self.env.default_action.astype(np.float32)
        new_obs, info = self.env.reset()
        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        rew = 0.0
        terminated, truncated = False, False
        if collect_samples:
            if self.crc_debug:
                info['crc_sample'] = (obs, act, new_obs, rew, terminated, truncated)
            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(act, new_obs, rew, terminated, truncated, info)
            else:
                sample = act, new_obs, rew, terminated, truncated, info
            self.buffer.append_sample(sample)
        return new_obs, info

    def step(self, obs, test, collect_samples, last_step=False):
        """
        Performs a full RL transition.

        A full RL transition is `obs` -> `act` -> `new_obs`, `rew`, `terminated`, `truncated`, `info`.
        Note that, in the Real-Time RL setting, `act` is appended to a buffer which is part of `new_obs`.
        This is because is does not directly affect the new observation, due to real-time delays.

        Args:
            obs (nested structure): previous observation
            test (bool): passed to the `act()` method of the `ActorModule`
            collect_samples (bool): if True, samples are buffered and sent to the `Server`
            last_step (bool): if True and `terminated` is False, `truncated` will be set to True

        Returns:
            new_obs (nested structure): new observation
            rew (float): new reward
            terminated (bool): episode termination signal
            truncated (bool): episode truncation signal
            info (dict): information dictionary
        """
        act = self.act(obs, test=test)
        new_obs, rew, terminated, truncated, info = self.env.step(act)
        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        if collect_samples:
            if last_step and not terminated:
                truncated = True
            if self.crc_debug:
                info['crc_sample'] = (obs, act, new_obs, rew, terminated, truncated)
            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(act, new_obs, rew, terminated, truncated, info)
            else:
                sample = act, new_obs, rew, terminated, truncated, info
            self.buffer.append_sample(sample)  # CAUTION: in the buffer, act is for the PREVIOUS transition (act, obs(act))
        return new_obs, rew, terminated, truncated, info

    def collect_train_episode(self, max_samples):
        """
        Collects a maximum of n training transitions (from reset to terminated or truncated)

        This method stores the episode and the train return in the local `Buffer` of the worker
        for sending to the `Server`.

        Args:
            max_samples (int): if the environment is not `terminated` after `max_samples` time steps,
                it is forcefully reset and `truncated` is set to True.
        """
        ret = 0.0
        steps = 0
        obs, info = self.reset(collect_samples=True)
        for i in range(max_samples):
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=False, collect_samples=True, last_step=i == max_samples - 1)
            ret += rew
            steps += 1
            if terminated or truncated:
                break
        self.buffer.stat_train_return = ret
        self.buffer.stat_train_steps = steps

    def run_episodes(self, max_samples_per_episode, nb_episodes=np.inf, train=False):
        """
        Runs `nb_episodes` episodes.

        Args:
            max_samples_per_episode (int): same as run_episode
            nb_episodes (int): total number of episodes to collect
            train (bool): same as run_episode
        """
        counter = 0
        while counter < nb_episodes:
            self.run_episode(max_samples_per_episode, train=train)
            counter += 1

    def run_episode(self, max_samples, train=False):
        """
        Collects a maximum of n test transitions (from reset to terminated or truncated).

        Args:
            max_samples (int): At most `max_samples` samples are collected per episode.
                If the episode is longer, it is forcefully reset and `truncated` is set to True.
            train (bool): whether the episode is a training or a test episode.
                `step` is called with `test=not train`.
        """
        ret = 0.0
        steps = 0
        obs, info = self.reset(collect_samples=False)
        for _ in range(max_samples):
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=not train, collect_samples=False)
            ret += rew
            steps += 1
            if terminated or truncated:
                break
        self.buffer.stat_test_return = ret
        self.buffer.stat_test_steps = steps

    def run(self, test_episode_interval=50, nb_episodes=np.inf):  # TODO: check number of collected samples are collected before sending
        """
        Runs the worker for `nb_episodes` episodes.

        This method is for training.
        It collects a test episode each `test_episode_interval` episodes.
        For deployment, use the `run_episodes` method instead.

        Args:
            test_episode_interval (int):
            nb_episodes (int):
        """
        episode = 0
        while episode < nb_episodes:
            if episode % test_episode_interval == 0 and not self.crc_debug:
                print_with_timestamp("running test episode")
                self.run_episode(self.max_samples_per_episode, train=False)
            print_with_timestamp("collecting train episode")
            self.collect_train_episode(self.max_samples_per_episode)
            print_with_timestamp("copying buffer for sending")
            self.send_and_clear_buffer()
            print_with_timestamp("checking for new weights")
            self.update_actor_weights()
            episode += 1
            # if self.crc_debug:
            #     break

    def profile_step(self):
        import torch.autograd.profiler as profiler
        obs, info = self.reset(collect_samples=True)
        use_cuda = True if self.device == 'cuda' else False
        print_with_timestamp(f"use_cuda:{use_cuda}")
        with profiler.profile(record_shapes=True, use_cuda=use_cuda) as prof:
            obs = collate([obs], device=self.device)
            with profiler.record_function("pytorch_profiler"):
                with torch.no_grad():
                    action_distribution = self.actor(obs)
                    action = action_distribution.sample()
        print_with_timestamp(prof.key_averages().table(row_limit=20, sort_by="cpu_time_total"))

    def run_env_benchmark(self, nb_steps, test=False):
        """
        Benchmarks the environment.

        This method is only compatible with rtgym_ environments.
        Furthermore, the `"benchmark"` option of the rtgym configuration dictionary must be set to `True`.

        .. _rtgym: https://github.com/yannbouteiller/rtgym

        Args:
            nb_steps (int): number of steps to perform to compute the benchmark
            test (int): whether the actor is called in test or train mode
        """
        obs, info = self.reset(collect_samples=False)
        for _ in range(nb_steps):
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=test, collect_samples=False)
            if terminated or truncated:
                obs, info = self.reset(collect_samples=False)
        print_with_timestamp(f"Benchmark results:\n{self.env.benchmarks()}")

    def send_and_clear_buffer(self):
        """
        Sends the buffered samples to the `Server`.
        """
        self.__buffer_lock.acquire()  # BUFFER LOCK.....................................................................
        self.__buffer += self.buffer
        self.__buffer_lock.release()  # END BUFFER LOCK.................................................................
        self.buffer.clear()

    def update_actor_weights(self):
        """
        Updates the actor with new weights received from the `Server` when available.
        """
        self.__weights_lock.acquire()  # WEIGHTS LOCK...................................................................
        if self.__weights is not None:  # new weights available
            with open(self.model_path, 'wb') as f:
                f.write(self.__weights)
            if self.model_history:
                self._cur_hist_cpt += 1
                if self._cur_hist_cpt == self.model_history:
                    x = datetime.datetime.now()
                    with open(self.model_path_history + str(x.strftime("%d_%m_%Y_%H_%M_%S")) + ".pth", 'wb') as f:
                        f.write(self.__weights)
                    self._cur_hist_cpt = 0
                    print_with_timestamp("model weights saved in history")
            self.actor.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print_with_timestamp("model weights have been updated")
            self.__weights = None
        self.__weights_lock.release()  # END WEIGHTS LOCK...............................................................
