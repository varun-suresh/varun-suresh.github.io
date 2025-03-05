
const pluginRss = require('@11ty/eleventy-plugin-rss')
const pluginNavigation = require('@11ty/eleventy-navigation')
const tufteWrapper = require('./util/tufteWrapper')
const linkToHead = require("./util/linkToHead");
const fs = require("node:fs")
const { DateTime } = require("luxon")

module.exports = function (eleventyConfig) {
	// export default function (eleventyConfig){
	// Plugins
	eleventyConfig.addPlugin(pluginRss)
	eleventyConfig.addPlugin(pluginNavigation)

	eleventyConfig.addFilter("linkToHead", linkToHead);

	// Asset Watch Targets
	eleventyConfig.addWatchTarget('./src/assets')

	/* Markdown Configuration */
	let options = {
		react: false,
	};

	// Markdown
	eleventyConfig.setLibrary("md", tufteWrapper)
	eleventyConfig.addFilter("markdown", tufteWrapper.render)
	eleventyConfig.addFilter("markdownInline", tufteWrapper.renderInline)


	// Layouts
	eleventyConfig.addLayoutAlias('base', 'base.njk')
	eleventyConfig.addLayoutAlias('simple', 'base.njk')
	eleventyConfig.addLayoutAlias('post', 'base.njk')
	eleventyConfig.addLayoutAlias('displaypage', 'displaypage.njk')


	// Pass-through files
	eleventyConfig.addPassthroughCopy('src/admin')
	eleventyConfig.addPassthroughCopy('src/assets')
	eleventyConfig.addPassthroughCopy('src/uploads')
	eleventyConfig.addPassthroughCopy('assets/img')
	// Deep-Merge
	eleventyConfig.setDataDeepMerge(true)
	// Date stuff
	eleventyConfig.addShortcode("year", () => `${new Date().getFullYear()}`); // useful for copyright
	eleventyConfig.addFilter("postDate", (dateObj) => {
		if (typeof dateObj === "string") {
			dateObj = DateTime.fromISO(dateObj);
		} else {
			dateObj = DateTime.fromJSDate(dateObj);
		}
		return dateObj.toFormat("MMMM d, yyyy");
	})
	eleventyConfig.addFilter("lastModifiedDate", function (filepath) {
		const stat = fs.statSync(filepath);
		return stat.mtime.toISOString();
	})


	// Base Config
	return {
		dir: {
			input: 'src',
			output: 'dist',
			includes: '_includes',
			layouts: '_layouts',
			data: '_data'
		},
		templateFormats: ['njk', 'md', '11ty.js'],
		htmlTemplateEngine: 'njk',
		markdownTemplateEngine: 'njk'
	}
}
