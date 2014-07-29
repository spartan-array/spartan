#include "base/all.h"

using namespace base;

TEST(strop, startswith) {
    EXPECT_TRUE(startswith("", ""));
    EXPECT_FALSE(startswith("", "a"));
}

TEST(strop, endswith) {
    EXPECT_TRUE(endswith("", ""));
    EXPECT_FALSE(endswith("", "a"));
}

TEST(strop, format_decimal) {
    EXPECT_EQ(format_decimal(-0.004999), "0.00");
    EXPECT_EQ(format_decimal(0.004), "0.00");
    EXPECT_EQ(format_decimal(-0.005), "-0.01");
    EXPECT_EQ(format_decimal(0.005), "0.01");
    EXPECT_EQ(format_decimal(0.0), "0.00");
    EXPECT_EQ(format_decimal(1000.0), "1,000.00");
    EXPECT_EQ(format_decimal(-1000.0), "-1,000.00");
    EXPECT_EQ(format_decimal(-12.345), "-12.35");
    EXPECT_EQ(format_decimal(12.345), "12.35");
    EXPECT_EQ(format_decimal(123.45), "123.45");
    EXPECT_EQ(format_decimal(1234.5), "1,234.50");
    EXPECT_EQ(format_decimal(1234567890.0), "1,234,567,890.00");
    EXPECT_EQ(format_decimal(123456789.0), "123,456,789.00");
    EXPECT_EQ(format_decimal(12345678.0), "12,345,678.00");
    EXPECT_EQ(format_decimal(-1234567890.0), "-1,234,567,890.00");
    EXPECT_EQ(format_decimal(-123456789.0), "-123,456,789.00");
    EXPECT_EQ(format_decimal(-12345678.0), "-12,345,678.00");

    EXPECT_EQ(format_decimal(-0), "0");
    EXPECT_EQ(format_decimal(0), "0");
    EXPECT_EQ(format_decimal(1000), "1,000");
    EXPECT_EQ(format_decimal(-1000), "-1,000");
    EXPECT_EQ(format_decimal(-12), "-12");
    EXPECT_EQ(format_decimal(12), "12");
    EXPECT_EQ(format_decimal(123), "123");
    EXPECT_EQ(format_decimal(1234), "1,234");
    EXPECT_EQ(format_decimal(1234567890), "1,234,567,890");
    EXPECT_EQ(format_decimal(123456789), "123,456,789");
    EXPECT_EQ(format_decimal(12345678), "12,345,678");
    EXPECT_EQ(format_decimal(-1234567890), "-1,234,567,890");
    EXPECT_EQ(format_decimal(-123456789), "-123,456,789");
    EXPECT_EQ(format_decimal(-12345678), "-12,345,678");
}

TEST(strop, strsplit) {
    std::vector<std::string>&& split = strsplit("hello world");
    EXPECT_EQ(split.size(), 2u);
    EXPECT_EQ(split[0], "hello");
    EXPECT_EQ(split[1], "world");

    split = strsplit("hello  world");
    EXPECT_EQ(split.size(), 2u);
    EXPECT_EQ(split[0], "hello");
    EXPECT_EQ(split[1], "world");

    split = strsplit(" hello  world");
    EXPECT_EQ(split.size(), 2u);
    EXPECT_EQ(split[0], "hello");
    EXPECT_EQ(split[1], "world");

    split = strsplit("hello  world ");
    EXPECT_EQ(split.size(), 2u);
    EXPECT_EQ(split[0], "hello");
    EXPECT_EQ(split[1], "world");

    split = strsplit("   hello  world   ");
    EXPECT_EQ(split.size(), 2u);
    EXPECT_EQ(split[0], "hello");
    EXPECT_EQ(split[1], "world");

    split = strsplit("hello/world", '/');
    EXPECT_EQ(split.size(), 2u);
    EXPECT_EQ(split[0], "hello");
    EXPECT_EQ(split[1], "world");

    split = strsplit("hello/world ", '/');
    EXPECT_EQ(split.size(), 2u);
    EXPECT_EQ(split[0], "hello");
    EXPECT_EQ(split[1], "world ");

    split = strsplit("      ");
    EXPECT_EQ(split.size(), 0u);

    split = strsplit("");
    EXPECT_EQ(split.size(), 0u);

    split = strsplit(" ");
    EXPECT_EQ(split.size(), 0u);

    split = strsplit("a");
    EXPECT_EQ(split.size(), 1u);
    EXPECT_EQ(split[0], "a");

    split = strsplit("a ");
    EXPECT_EQ(split.size(), 1u);
    EXPECT_EQ(split[0], "a");

    split = strsplit(" a");
    EXPECT_EQ(split.size(), 1u);
    EXPECT_EQ(split[0], "a");

    split = strsplit(" a ");
    EXPECT_EQ(split.size(), 1u);
    EXPECT_EQ(split[0], "a");
}

