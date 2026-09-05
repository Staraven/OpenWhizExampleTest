/*
* gCanvas.cpp
*
*  Created on: May 6, 2020
*      Author: Noyan Culum
*/


#include "gCanvas.h"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "OpenWhiz/text/owLanguage.hpp"
#include "OpenWhiz/text/owStemmer.hpp"
#include "OpenWhiz/text/owEmbeddingLookup.hpp"
#include "OpenWhiz/text/owSentimentPreset.hpp"
#include "OpenWhiz/text/tokenizers/owTextTokenizer.hpp"
#include "OpenWhiz/text/owClusterLabeler.hpp"
#include "OpenWhiz/text/owTfIdfVectorizer.hpp"

namespace {

// ===== Port of libs/OpenWhiz/examples/textClassificationExample/main.cpp =====
// Same pipeline/data/logic as the standalone example, only the I/O changed:
// std::cout -> gLogi, and file paths -> gApp::gGetFilesDir() (GlistEngine's runtime
// asset/writable directory) instead of paths relative to the OpenWhiz repo root.

struct LangSample {
	ow::owLanguage language;
	std::string name;
	std::string vecFile;
	std::vector<std::pair<std::string, std::vector<float>>> words;
	std::vector<std::pair<std::string, int>> trainSentences; // text, label (0=animal, 1=vehicle)
	std::vector<std::string> testSentences;
};

void writeVecFile(const std::string& path, const std::vector<std::pair<std::string, std::vector<float>>>& words) {
	std::ofstream f(path);
	f << words.size() << " " << words.front().second.size() << "\n";
	for (auto& wv : words) {
		f << wv.first;
		for (float v : wv.second) f << " " << v;
		f << "\n";
	}
}

void runLanguageDemo(const LangSample& sample) {
	gLogi("TextClassificationExample") << "=== " << sample.name << " ===";
	writeVecFile(sample.vecFile, sample.words);

	ow::owTextTokenizer tokenizer(sample.language);
	ow::owStemmer stemmer(sample.language);
	ow::owEmbeddingLookup lookup;
	if (!lookup.loadFromFile(sample.vecFile)) {
		gLogi("TextClassificationExample") << "failed to load " << sample.vecFile;
		return;
	}

	auto embed = [&](const std::string& text) {
		std::vector<std::string> tokens = tokenizer.tokenize(text);
		std::vector<std::string> stemmed;
		for (auto& t : tokens) stemmed.push_back(stemmer.stem(t));
		return lookup.embedAverage(stemmed);
	};

	std::vector<std::vector<float>> trainEmb;
	std::vector<int> trainLabels;
	for (auto& ts : sample.trainSentences) {
		trainEmb.push_back(embed(ts.first));
		trainLabels.push_back(ts.second);
	}

	ow::owSentimentPreset::Options options;
	options.hiddenSizes = {4};
	options.maxEpochs = 100;
	// All 8 synthetic examples are used for training - this is a mechanism demo,
	// not a real held-out evaluation.
	options.trainRatio = 1.0f;
	options.valRatio = 0.0f;
	options.testRatio = 0.0f;
	// Fixed seed so the demo's predictions are the same every run - see
	// owSentimentPreset.hpp's Options::useSeed comment for why this exists.
	options.useSeed = true;
	options.seed = 42;

	ow::owSentimentPreset classifier;
	std::string tempCsv = sample.vecFile + ".train.csv";
	if (!classifier.train(trainEmb, trainLabels, 2, tempCsv, options)) {
		gLogi("TextClassificationExample") << "training failed";
		return;
	}

	for (auto& text : sample.testSentences) {
		auto scores = classifier.predict(embed(text));
		std::string predicted = scores[1] > scores[0] ? "vehicle" : "animal";
		gLogi("TextClassificationExample") << "\"" << text << "\" -> " << predicted
			<< " (P(animal)=" << scores[0] << " P(vehicle)=" << scores[1] << ")";
	}
}

void runTextClassificationExample() {
	gLogi("TextClassificationExample") << "=== OpenWhiz/text: tokenize -> stem -> embed -> classify (EN/TR/FR) ===";

	LangSample english{
		ow::owLanguage::English, "English", gApp::gGetFilesDir() + "en.vec",
		{
			{"cat", {1.0f, 0.0f, 0.0f, 0.0f}}, {"dog", {0.9f, 0.1f, 0.0f, 0.0f}},
			{"bird", {0.8f, 0.2f, 0.0f, 0.0f}}, {"pet", {0.85f, 0.15f, 0.0f, 0.0f}},
			{"car", {0.0f, 0.0f, 1.0f, 0.0f}}, {"bus", {0.0f, 0.0f, 0.9f, 0.1f}},
			{"train", {0.0f, 0.0f, 0.8f, 0.2f}}, {"drive", {0.0f, 0.0f, 0.85f, 0.15f}},
		},
		{
			{"I have a cat", 0}, {"My dog is happy", 0}, {"The bird sings", 0}, {"I love my pet", 0},
			{"I drive my car", 1}, {"The bus is late", 1}, {"We took the train", 1}, {"I like to drive", 1},
		},
		{"My cat and dog are friends", "The train and bus were both late"}
	};

	LangSample turkish{
		ow::owLanguage::Turkish, "Turkish", gApp::gGetFilesDir() + "tr.vec",
		{
			{"kedi", {1.0f, 0.0f, 0.0f, 0.0f}}, {"köpek", {0.9f, 0.1f, 0.0f, 0.0f}},
			{"kuş", {0.8f, 0.2f, 0.0f, 0.0f}}, {"hayvan", {0.85f, 0.15f, 0.0f, 0.0f}},
			{"araba", {0.0f, 0.0f, 1.0f, 0.0f}}, {"otobüs", {0.0f, 0.0f, 0.9f, 0.1f}},
			{"tren", {0.0f, 0.0f, 0.8f, 0.2f}}, {"sürmek", {0.0f, 0.0f, 0.85f, 0.15f}},
		},
		{
			{"Kediler çok tatlı", 0}, {"Köpekler mutlu", 0}, {"Kuşlar öter", 0}, {"Hayvanlar güzel", 0},
			{"Arabalar hızlı", 1}, {"Otobüsler geç kaldı", 1}, {"Trenler rahat", 1}, {"Sürmek güzel", 1},
		},
		{"Kediler ve köpekler arkadaş", "Otobüs ve tren geç kaldı"}
	};

	LangSample french{
		ow::owLanguage::French, "French", gApp::gGetFilesDir() + "fr.vec",
		{
			{"chat", {1.0f, 0.0f, 0.0f, 0.0f}}, {"chien", {0.9f, 0.1f, 0.0f, 0.0f}},
			{"oiseau", {0.8f, 0.2f, 0.0f, 0.0f}}, {"animal", {0.85f, 0.15f, 0.0f, 0.0f}},
			{"voiture", {0.0f, 0.0f, 1.0f, 0.0f}}, {"bus", {0.0f, 0.0f, 0.9f, 0.1f}},
			{"train", {0.0f, 0.0f, 0.8f, 0.2f}}, {"conduire", {0.0f, 0.0f, 0.85f, 0.15f}},
		},
		{
			{"J'ai un chat", 0}, {"Mon chien est content", 0}, {"L'oiseau chante", 0}, {"Cet animal est mignon", 0},
			{"Je conduis ma voiture", 1}, {"Le bus est en retard", 1}, {"Nous avons pris le train", 1}, {"J'aime conduire", 1},
		},
		{"Mon chat et mon chien sont amis", "Le train et le bus étaient en retard"}
	};

	runLanguageDemo(english);
	runLanguageDemo(turkish);
	runLanguageDemo(french);
}

// ===== Port of libs/OpenWhiz/examples/clusterLabelingExample/main.cpp =====
// Same pipeline/data/logic as the standalone example, only the I/O changed:
// std::cout -> gLogi, and the input file path -> gApp::gGetFilesDir() + "twenty_ng_sample.txt"
// (copy that file into this project's assets/files/ - see setup instructions).

static const char* kLabelNames[7] = {
	"", "Hockey", "Space", "Medicine", "MideastPolitics", "Graphics", "ForSale"
};

struct LabeledDoc {
	int trueLabel = 0;  // 1..6, see kLabelNames
	std::string title;
	std::string description;
};

std::vector<LabeledDoc> loadLabeledSample(const std::string& path) {
	std::vector<LabeledDoc> docs;
	std::ifstream file(path);
	std::string line;
	while (std::getline(file, line)) {
		if (line.empty()) continue;
		size_t p1 = line.find("|||");
		size_t p2 = (p1 == std::string::npos) ? std::string::npos : line.find("|||", p1 + 3);
		if (p1 == std::string::npos || p2 == std::string::npos) continue;
		LabeledDoc doc;
		doc.trueLabel = std::stoi(line.substr(0, p1));
		doc.title = line.substr(p1 + 3, p2 - (p1 + 3));
		doc.description = line.substr(p2 + 3);
		docs.push_back(doc);
	}
	return docs;
}

// owClusterLayer seeds centroids from a non-seedable std::random_device, so
// this builds an XML string with the same [-1,1] uniform distribution but a
// fixed-seed RNG, then loads it via owClusterLayer's public fromXML() right
// after construction - reproducing its default init with a fixed seed,
// without modifying owClusterLayer.hpp.
std::string buildSeededInitialCentroidsXml(int numClusters, size_t dim, unsigned int seed) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::ostringstream values;
	for (int c = 0; c < numClusters; ++c) {
		for (size_t d = 0; d < dim; ++d) {
			values << dist(rng);
			if (!(c == numClusters - 1 && d == dim - 1)) values << ' ';
		}
	}
	std::ostringstream xml;
	xml << "<InputSize>" << dim << "</InputSize>"
		<< "<NumClusters>" << numClusters << "</NumClusters>"
		<< "<Centroids>" << values.str() << "</Centroids>";
	return xml.str();
}

void runClusterLabelingExample() {
	gLogi("ClusterLabelingExample") << "=== OpenWhiz/text: cluster labeling mechanism on 20 Newsgroups (public benchmark) ===";

	std::vector<LabeledDoc> docs = loadLabeledSample(gApp::gGetFilesDir() + "twenty_ng_sample.txt");
	if (docs.empty()) {
		gLogi("ClusterLabelingExample") << "Failed to load twenty_ng_sample.txt - copy it into assets/files/.";
		return;
	}
	gLogi("ClusterLabelingExample") << "Loaded " << docs.size() << " 20 Newsgroups rows (Hockey/Space/Medicine/"
		"MideastPolitics/Graphics/ForSale, 50 each).";

	ow::owTextTokenizer tokenizer(ow::owLanguage::English);
	ow::owStemmer stemmer(ow::owLanguage::English);  // no-op for English, kept for pipeline consistency

	// A short list of generic English function words - without it, "the",
	// "of", "said" etc. dominate every cluster's bag-of-stems label instead of
	// actual content words.
	static const std::unordered_set<std::string> kStopwords = {
		"the", "a", "an", "of", "to", "in", "on", "for", "and", "is", "are",
		"was", "were", "be", "been", "with", "by", "at", "as", "from", "that",
		"this", "it", "its", "has", "have", "had", "will", "after", "over",
		"new", "said", "not", "but", "or", "his", "her", "their", "than",
		"into", "up", "out", "about", "who", "which", "he", "she", "they",
		"i", "you", "we", "my", "your", "if", "so", "do", "does", "did",
		"can", "could", "would", "should", "one", "would", "also",
	};

	std::vector<std::vector<std::string>> stemLists;
	stemLists.reserve(docs.size());
	for (const LabeledDoc& doc : docs) {
		std::vector<std::string> tokens = tokenizer.tokenize(doc.title + " " + doc.description);
		std::vector<std::string> stems;
		stems.reserve(tokens.size());
		for (const std::string& tok : tokens) {
			if (kStopwords.count(tok) > 0) continue;
			std::string stemmed = stemmer.stem(tok);
			if (kStopwords.count(stemmed) > 0) continue;
			stems.push_back(stemmed);
		}
		stemLists.push_back(stems);
	}

	// Ranking vocabulary by raw document frequency alone favors generic
	// filler over discriminative terms just below the cap, so
	// maxDocFrequencyRatio excludes near-universal terms first.
	ow::owTfIdfVectorizer vectorizer;
	ow::owTfIdfVectorizer::Options vecOptions;
	vecOptions.minDocFrequency = 2;
	vecOptions.maxDocFrequencyRatio = 0.5f;
	vecOptions.maxVocabularySize = 3000;
	vecOptions.l2Normalize = true;
	std::vector<std::vector<float>> tfidfVectors = vectorizer.fitTransform(stemLists, vecOptions);
	gLogi("ClusterLabelingExample") << "TF-IDF vocabulary size: " << vectorizer.vocabulary().size();

	const int numClusters = 6;  // matches this sample's 6 known classes
	const size_t dim = tfidfVectors[0].size();
	const int maxEpochs = 200;
	const float learningRate = 0.05f;

	// Reproduces owClusterLabeler::cluster()'s training loop (same layer,
	// loss, optimizer, objective) inline so a seeded centroid set can be
	// injected before training - owClusterLabeler.hpp has no seed parameter
	// and is not modified here.
	ow::owClusterLayer layer(dim, static_cast<size_t>(numClusters));
	layer.fromXML(buildSeededInitialCentroidsXml(numClusters, dim, /*seed=*/42));

	ow::owADAMOptimizer optimizer(learningRate);
	layer.setOptimizer(&optimizer);
	ow::owMeanSquaredErrorLoss loss;

	ow::owTensor<float, 2> input(tfidfVectors.size(), dim);
	for (size_t i = 0; i < tfidfVectors.size(); ++i) {
		for (size_t d = 0; d < dim; ++d) input(i, d) = tfidfVectors[i][d];
	}
	ow::owTensor<float, 2> target(tfidfVectors.size(), static_cast<size_t>(numClusters));
	target.setZero();

	for (int epoch = 0; epoch < maxEpochs; ++epoch) {
		ow::owTensor<float, 2> output = layer.forward(input);
		ow::owTensor<float, 2> grad = loss.gradient(output, target);
		layer.backward(grad);
		layer.train();
	}

	ow::owTensor<float, 2> finalDistances = layer.forward(input);
	std::vector<int> assignments(tfidfVectors.size());
	for (size_t i = 0; i < tfidfVectors.size(); ++i) {
		int best = 0;
		float bestDist = finalDistances(i, 0);
		for (int c = 1; c < numClusters; ++c) {
			if (finalDistances(i, c) < bestDist) { bestDist = finalDistances(i, c); best = c; }
		}
		assignments[i] = best;
	}

	const int numTrueLabels = 6;

	int totalMajorityCorrect = 0;
	for (int c = 0; c < numClusters; ++c) {
		std::unordered_map<std::string, int> stemFreq;
		std::vector<int> counts(numTrueLabels + 1, 0);
		int size = 0;
		for (size_t i = 0; i < assignments.size(); ++i) {
			if (assignments[i] != c) continue;
			++size;
			counts[docs[i].trueLabel]++;
			for (const std::string& stem : stemLists[i]) stemFreq[stem]++;
		}
		std::vector<std::pair<std::string, int>> topStems(stemFreq.begin(), stemFreq.end());
		std::sort(topStems.begin(), topStems.end(),
				  [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) {
					  return a.second > b.second;
				  });
		if (topStems.size() > 8) topStems.resize(8);

		int majorityLabel = 1;
		for (int lbl = 2; lbl <= numTrueLabels; ++lbl) {
			if (counts[lbl] > counts[majorityLabel]) majorityLabel = lbl;
		}
		totalMajorityCorrect += counts[majorityLabel];

		std::string topStemsStr;
		for (const auto& sf : topStems) topStemsStr += " " + sf.first + "(" + std::to_string(sf.second) + ")";
		gLogi("ClusterLabelingExample") << "Cluster " << c << " (n=" << size << ") top stems:" << topStemsStr;

		std::string breakdown;
		for (int lbl = 1; lbl <= numTrueLabels; ++lbl) {
			breakdown += std::string(" ") + kLabelNames[lbl] + "=" + std::to_string(counts[lbl]);
		}
		gLogi("ClusterLabelingExample") << " " << breakdown << " (majority=" << kLabelNames[majorityLabel] << ")";
	}

	float purity = 100.0f * totalMajorityCorrect / docs.size();
	gLogi("ClusterLabelingExample") << "Purity: " << purity << "% (chance baseline for 6 balanced classes: ~16.7%), "
		<< totalMajorityCorrect << "/" << docs.size() << " documents in their cluster's majority class";
}

} // namespace


gCanvas::gCanvas(gApp* root) : gBaseCanvas(root) {
	this->root = root;
}

gCanvas::~gCanvas() {
}

void gCanvas::setup() {
	logo.loadImage("glistengine_logo.png");

	runTextClassificationExample();
	runClusterLabelingExample();
}

void gCanvas::update() {
}

void gCanvas::draw() {
	logo.draw((getWidth() - logo.getWidth()) / 2, (getHeight() - logo.getHeight()) / 2);
}

void gCanvas::keyPressed(int key) {
//	gLogi("gCanvas") << "keyPressed:" << key;
}

void gCanvas::keyReleased(int key) {
//	gLogi("gCanvas") << "keyReleased:" << key;
}

void gCanvas::charPressed(unsigned int codepoint) {
//	gLogi("gCanvas") << "charPressed:" << gCodepointToStr(codepoint);
}

void gCanvas::mouseMoved(int x, int y) {
//	gLogi("gCanvas") << "mouseMoved" << ", x:" << x << ", y:" << y;
}

void gCanvas::mouseDragged(int x, int y, int button) {
//	gLogi("gCanvas") << "mouseDragged" << ", x:" << x << ", y:" << y << ", b:" << button;
}

void gCanvas::mousePressed(int x, int y, int button) {
//	gLogi("gCanvas") << "mousePressed" << ", x:" << x << ", y:" << y << ", b:" << button;
}

void gCanvas::mouseReleased(int x, int y, int button) {
//	gLogi("gCanvas") << "mouseReleased" << ", button:" << button;
}

void gCanvas::mouseScrolled(int x, int y) {
//	gLogi("gCanvas") << "mouseScrolled" << ", x:" << x << ", y:" << y;
}

void gCanvas::mouseEntered() {

}

void gCanvas::mouseExited() {

}

void gCanvas::windowResized(int w, int h) {

}

void gCanvas::showNotify() {

}

void gCanvas::hideNotify() {

}
