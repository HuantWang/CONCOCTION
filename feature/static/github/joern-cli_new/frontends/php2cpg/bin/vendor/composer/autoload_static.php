<?php

// autoload_static.php @generated by Composer

namespace Composer\Autoload;

class ComposerStaticInitf3f249c1997cd8080cd00bda37de55b8
{
    public static $prefixLengthsPsr4 = array (
        'P' => 
        array (
            'PhpParser\\' => 10,
        ),
    );

    public static $prefixDirsPsr4 = array (
        'PhpParser\\' => 
        array (
            0 => __DIR__ . '/..' . '/nikic/php-parser/lib/PhpParser',
        ),
    );

    public static $classMap = array (
        'Composer\\InstalledVersions' => __DIR__ . '/..' . '/composer/InstalledVersions.php',
    );

    public static function getInitializer(ClassLoader $loader)
    {
        return \Closure::bind(function () use ($loader) {
            $loader->prefixLengthsPsr4 = ComposerStaticInitf3f249c1997cd8080cd00bda37de55b8::$prefixLengthsPsr4;
            $loader->prefixDirsPsr4 = ComposerStaticInitf3f249c1997cd8080cd00bda37de55b8::$prefixDirsPsr4;
            $loader->classMap = ComposerStaticInitf3f249c1997cd8080cd00bda37de55b8::$classMap;

        }, null, ClassLoader::class);
    }
}
