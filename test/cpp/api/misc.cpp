#include <catch.hpp>

#include <torch/detail/ordered_dict.h>
#include <torch/expanding_array.h>
#include <torch/torch.h>

using namespace torch;
using namespace torch::nn;
using namespace torch::detail;

using Catch::StartsWith;

TEST_CASE("misc") {
  SECTION("no_grad") {
    no_grad_guard guard;
    auto model = Linear(5, 2).build();
    auto x = Var(at::CPU(at::kFloat).randn({10, 5}), true);
    auto y = model->forward({x})[0];
    Variable s = y.sum();

    backward(s);
    REQUIRE(!model->parameters2().get("weight").grad().defined());
  }

  SECTION("CPU random seed") {
    int size = 100;
    setSeed(7);
    auto x1 = Var(at::CPU(at::kFloat).randn({size}));
    setSeed(7);
    auto x2 = Var(at::CPU(at::kFloat).randn({size}));

    auto l_inf = (x1.data() - x2.data()).abs().max().toCFloat();
    REQUIRE(l_inf < 1e-10);
  }
}

TEST_CASE("misc_cuda", "[cuda]") {
  SECTION("CUDA random seed") {
    int size = 100;
    setSeed(7);
    auto x1 = Var(at::CUDA(at::kFloat).randn({size}));
    setSeed(7);
    auto x2 = Var(at::CUDA(at::kFloat).randn({size}));

    auto l_inf = (x1.data() - x2.data()).abs().max().toCFloat();
    REQUIRE(l_inf < 1e-10);
  }
}

TEST_CASE("expanding-array") {
  SECTION("successful construction") {
    SECTION("initializer_list") {
      ExpandingArray<5> e({1, 2, 3, 4, 5});
      REQUIRE(e.size() == 5);
      for (size_t i = 0; i < e.size(); ++i) {
        REQUIRE((*e)[i] == i + 1);
      }
    }

    SECTION("vector") {
      ExpandingArray<5> e(std::vector<int64_t>{1, 2, 3, 4, 5});
      REQUIRE(e.size() == 5);
      for (size_t i = 0; i < e.size(); ++i) {
        REQUIRE((*e)[i] == i + 1);
      }
    }

    SECTION("array") {
      ExpandingArray<5> e(std::array<int64_t, 5>({1, 2, 3, 4, 5}));
      REQUIRE(e.size() == 5);
      for (size_t i = 0; i < e.size(); ++i) {
        REQUIRE((*e)[i] == i + 1);
      }
    }

    SECTION("single value") {
      ExpandingArray<5> e(5);
      REQUIRE(e.size() == 5);
      for (size_t i = 0; i < e.size(); ++i) {
        REQUIRE((*e)[i] == 5);
      }
    }
  }
  SECTION("throws for incorrect size on construction") {
    SECTION("initializer_list") {
      REQUIRE_THROWS_WITH(
          ExpandingArray<5>({1, 2, 3, 4, 5, 6, 7}),
          StartsWith("Expected 5 values, but instead got 7"));
    }
    SECTION("vector") {
      REQUIRE_THROWS_WITH(
          ExpandingArray<5>(std::vector<int64_t>({1, 2, 3, 4, 5, 6, 7})),
          StartsWith("Expected 5 values, but instead got 7"));
    }
  }
}

TEST_CASE("ordered-dict") {
  SECTION("is empty after default construction") {
    OrderedDict<int> dict;
    REQUIRE(dict.is_empty());
    REQUIRE(dict.size() == 0);
  }

  SECTION("insert inserts elements when they are not yet present") {
    OrderedDict<int> dict;
    dict.insert("a", 1);
    dict.insert("b", 2);
    REQUIRE(dict.size() == 2);
  }

  SECTION("get returns values when present") {
    OrderedDict<int> dict;
    dict.insert("a", 1);
    dict.insert("b", 2);
    REQUIRE(dict.get("a") == 1);
    REQUIRE(dict.get("b") == 2);
  }

  SECTION("get throws when passed keys that are not present") {
    OrderedDict<int> dict;
    dict.insert("a", 1);
    dict.insert("b", 2);
    REQUIRE_THROWS_WITH(dict.get("foo"), StartsWith("No such key: 'foo'"));
    REQUIRE_THROWS_WITH(dict.get(""), StartsWith("No such key: ''"));
  }

  SECTION("can initialize from list") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE(dict.size() == 2);
    REQUIRE(dict.get("a") == 1);
    REQUIRE(dict.get("b") == 2);
  }

  SECTION("insert throws when passed elements that are present") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE_THROWS_WITH(
        dict.insert("a", 1), StartsWith("Key 'a' already present"));
    REQUIRE_THROWS_WITH(
        dict.insert("b", 1), StartsWith("Key 'b' already present"));
  }

  SECTION("front() returns the first item") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE(dict.front().key == "a");
    REQUIRE(dict.front().value == 1);
  }

  SECTION("back() returns the last item") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE(dict.back().key == "b");
    REQUIRE(dict.back().value == 2);
  }

  SECTION("find returns pointers to values when present") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE(dict.find("a") != nullptr);
    REQUIRE(*dict.find("a") == 1);
    REQUIRE(dict.find("b") != nullptr);
    REQUIRE(*dict.find("b") == 2);
  }

  SECTION("find returns null pointers when passed keys that are not present") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE(dict.find("bar") == nullptr);
    REQUIRE(dict.find("") == nullptr);
  }

  SECTION("operator[] returns values when passed keys that are present") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE(dict["a"] == 1);
    REQUIRE(dict["b"] == 2);
  }

  SECTION("operator[] returns items positionally when passed integers") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE(dict[0].key == "a");
    REQUIRE(dict[0].value == 1);
    REQUIRE(dict[1].key == "b");
    REQUIRE(dict[1].value == 2);
  }

  SECTION("operator[] throws when passed keys that are not present") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE_THROWS_WITH(dict.get("foo"), StartsWith("No such key: 'foo'"));
    REQUIRE_THROWS_WITH(dict.get(""), StartsWith("No such key: ''"));
  }

  SECTION("update inserts all items from another OrderedDict") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    OrderedDict<int> dict2 = {{"c", 3}};
    dict2.update(dict);
    REQUIRE(dict2.size() == 3);
    REQUIRE(dict2.find("a") != nullptr);
    REQUIRE(dict2.find("b") != nullptr);
    REQUIRE(dict2.find("c") != nullptr);
  }

  SECTION("update also checks for duplicates") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    OrderedDict<int> dict2 = {{"a", 1}};
    REQUIRE_THROWS_WITH(
        dict2.update(dict), StartsWith("Key 'a' already present"));
  }

  SECTION("Can iterate items") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    auto iterator = dict.begin();
    REQUIRE(iterator != dict.end());
    REQUIRE(iterator->key == "a");
    REQUIRE(iterator->value == 1);
    ++iterator;
    REQUIRE(iterator != dict.end());
    REQUIRE(iterator->key == "b");
    REQUIRE(iterator->value == 2);
    ++iterator;
    REQUIRE(iterator == dict.end());
  }

  SECTION("clear makes the dict empty") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE(!dict.is_empty());
    dict.clear();
    REQUIRE(dict.is_empty());
  }

  SECTION("can copy construct") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    OrderedDict<int> copy = dict;
    REQUIRE(copy.size() == 2);
    REQUIRE(*copy[0] == 1);
    REQUIRE(*copy[1] == 2);
  }

  SECTION("can copy assign") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    OrderedDict<int> copy = {{"c", 1}};
    REQUIRE(copy.find("c") != nullptr);
    copy = dict;
    REQUIRE(copy.size() == 2);
    REQUIRE(*copy[0] == 1);
    REQUIRE(*copy[1] == 2);
    REQUIRE(copy.find("c") == nullptr);
  }

  SECTION("can move construct") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    OrderedDict<int> copy = std::move(dict);
    REQUIRE(copy.size() == 2);
    REQUIRE(*copy[0] == 1);
    REQUIRE(*copy[1] == 2);
  }

  SECTION("can move assign") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    OrderedDict<int> copy = {{"c", 1}};
    REQUIRE(copy.find("c") != nullptr);
    copy = std::move(dict);
    REQUIRE(copy.size() == 2);
    REQUIRE(*copy[0] == 1);
    REQUIRE(*copy[1] == 2);
    REQUIRE(copy.find("c") == nullptr);
  }

  SECTION("can insert with braces") {
    OrderedDict<std::pair<int, int>> dict;
    dict.insert("a", {1, 2});
    REQUIRE(!dict.is_empty());
    REQUIRE(dict["a"].first == 1);
    REQUIRE(dict["a"].second == 2);
  }
}
