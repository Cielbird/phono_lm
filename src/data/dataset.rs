use burn::data::dataset::{
    Dataset, SqliteDataset,
    source::huggingface::HuggingfaceDatasetLoader,
    transform::PartialDataset,
};

#[derive(new, Clone, Debug)]
pub struct PhonoGenerationItem {
    pub text: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IpaChildesItem {
    pub ipa_transcription: String,
}

pub struct IpaChildesDataset {
    dataset: PartialDataset<SqliteDataset<IpaChildesItem>, IpaChildesItem>,
}

impl Dataset<PhonoGenerationItem> for IpaChildesDataset {
    fn get(&self, index: usize) -> Option<PhonoGenerationItem> {
        self.dataset
            .get(index)
            .map(|item| PhonoGenerationItem::new(item.ipa_transcription))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl IpaChildesDataset {
    pub fn train() -> Self {
        let dataset = Self::open();
        let n = dataset.len();
        Self { dataset: PartialDataset::new(dataset, 0, n * 4 / 5) }
    }

    pub fn test() -> Self {
        let dataset = Self::open();
        let n = dataset.len();
        Self { dataset: PartialDataset::new(dataset, n * 4 / 5, n) }
    }

    fn open() -> SqliteDataset<IpaChildesItem> {
        HuggingfaceDatasetLoader::new("phonemetransformers/IPA-CHILDES")
            .with_subset("EnglishNA")
            .dataset("train")
            .unwrap()
    }
}
