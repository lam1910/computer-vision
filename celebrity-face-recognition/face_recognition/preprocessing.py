from facenet_pytorch import MTCNN
from face_recognition.data_manager import create_cropped_face_dataset
from config.config import Config


def crop_face(cfg: Config):
    mtcnn = MTCNN(
        image_size=cfg.image_size,
        margin=cfg.margin,
        min_face_size=cfg.min_face_size,
        thresholds=cfg.threshold,
        factor=cfg.factor,
        # prewhiten=cfg.prewhiten,
        keep_all=True,
        device=cfg.device
    )

    create_cropped_face_dataset(mtcnn,
                                cfg.image_size,
                                cfg.batch_size,
                                cfg.num_workers,
                                cfg.pin_memory)

    del mtcnn


if __name__ == '__main__':
    cfg = Config()
    crop_face(cfg)
