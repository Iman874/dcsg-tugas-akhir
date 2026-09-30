# Referensi Ilmiah

Daftar pustaka penelitian Distance-Controlled Skip-Gram (DCSG), dirapikan dari dua artikel jurnal:
*Perancangan Arsitektur Distance-Controlled Skip-Gram untuk Representasi Teks Bahasa Indonesia* dan
*Distance-Controlled Skip-Gram untuk Representasi Teks Bahasa Indonesia (Evaluasi vs Word2Vec)*.

## Model embedding dasar

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv:1301.3781. https://arxiv.org/abs/1301.3781
2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. Advances in Neural Information Processing Systems 26, 3111-3119. https://arxiv.org/abs/1310.4546
3. Levy, O., & Goldberg, Y. (2014). *Neural Word Embedding as Implicit Matrix Factorization*. Advances in Neural Information Processing Systems 27, 2177-2185.
4. Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation*. EMNLP 2014, 1532-1543. https://nlp.stanford.edu/pubs/glove.pdf
5. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems 30, 5998-6008. https://arxiv.org/abs/1706.03762
6. Finkelstein, L., Gabrilovich, E., Matias, Y., Rivlin, E., Solan, Z., Wolfman, G., & Ruppin, E. (2001). *Placing Search in Context: The Concept Revisited*. Proceedings of the 10th International World Wide Web Conference, 116-131. https://doi.org/10.1145/371920.371981

## NLP bahasa Indonesia

7. Purwarianti, A., & Crisdayanti, I. A. P. A. (2019). *Improving Bi-LSTM Performance for Indonesian Sentiment Analysis Paragraph-Level*. 2019 International Conference on Advanced Informatics: Concepts, Theory and Applications (ICAICTA), 1-5. IEEE.
8. Koto, F., Rahimi, A., Lau, J. H., & Baldwin, T. (2020). *IndoLEM and IndoBERT: A Benchmark Dataset and Pre-trained Language Model for Indonesian NLP*. COLING 2020, 757-770. https://aclanthology.org/2020.coling-main.66/
9. Wilie, B., Vincentio, K., Winata, H. S., Remzi, B., Li, M.-J., Fung, P., Koto, F., Baharuddin, M., & Purwarianti, A. (2020). *IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding*. Findings of the 1st Conference of the Asia-Pacific Chapter of the ACL and the 10th IJNLP, 843-857. https://arxiv.org/abs/2009.05387

## Sumber daya dan benchmark yang dipakai

Nama resource di bawah ini dipakai dalam evaluasi; tautan resminya sengaja tidak dicantumkan karena perlu diverifikasi ulang sebelum dipublikasi.

10. WordSim-353 versi bahasa Indonesia — pasangan kata berlabel kemiripan, sumber metrik kosinus dan Spearman.
11. Google Analogy Dataset — 19.544 soal analogi kata, sumber metrik top-1/top-5.
12. SmSA (Sentiment Analisa) — dataset sentimen bahasa Indonesia dari IndoNLU, dipakai dengan Logistic Regression.
13. IndoLEM — benchmark dataset dan pre-trained model bahasa Indonesia.
14. SemRel 2024 — shared task relasi semantik lintas bahasa, termasuk bahasa Indonesia (bahan pembanding metodologi).
15. KBBI — kosakata baku bahasa Indonesia, dipakai sebagai filter leksikal pada preprocessing.
16. Gensim — implementasi Word2Vec sebagai baseline. https://radimrehurek.com/gensim/

## Catatan

- Rujukan 10-16 adalah resource yang dipakai ulang, bukan hasil penelitian ini; masing-masing punya lisensi sendiri dan tidak disalin ke repo ini.
- Korpus pelatihan: dump Wikipedia Bahasa Indonesia 2023 (lisensi CC BY-SA), https://dumps.wikimedia.org.
