from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'celebs_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['celebaMask_train_label'],
		'train_target_root': dataset_paths['celebaMask_train'],
		'test_source_root': dataset_paths['celebaMask_test_label'],
		'test_target_root': dataset_paths['celebaMask_test'],
	},
	'celebs_landmark_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['celebaLandmark_train_label'],
		'train_target_root': dataset_paths['celebaLandmark_train'],
		'test_source_root': dataset_paths['celebaLandmark_test_label'],
		'test_target_root': dataset_paths['celebaLandmark_test'],
	},
	'lsunchurch_seg_to_img': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['lsunchurch_train_label'],
		'train_target_root': dataset_paths['lsunchurch_train'],
		'test_source_root': dataset_paths['lsunchurch_test_label'],
		'test_target_root': dataset_paths['lsunchurch_test'],
		},
	'lsuncar_seg_to_img': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['lsuncar_train_label'],
		'train_target_root': dataset_paths['lsuncar_train'],
		'test_source_root': dataset_paths['lsuncar_test_label'],
		'test_target_root': dataset_paths['lsuncar_test'],
		},
	'lsuncat_scribble_to_img': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['lsuncat_train_label'],
		'train_target_root': dataset_paths['lsuncat_train'],
		'test_source_root': dataset_paths['lsuncat_test_label'],
		'test_target_root': dataset_paths['lsuncat_test'],
		},
	'ukiyo-e_scribble_to_img': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['ukiyo-e_train_label'],
		'train_target_root': dataset_paths['ukiyo-e_train'],
		'test_source_root': dataset_paths['ukiyo-e_test_label'],
		'test_target_root': dataset_paths['ukiyo-e_test'],
		},
	'anime_cross_to_img': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['anime_train_label'],
		'train_target_root': dataset_paths['anime_train'],
		'test_source_root': dataset_paths['anime_test_label'],
		'test_target_root': dataset_paths['anime_test'],
		}
}
